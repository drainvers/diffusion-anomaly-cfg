"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse, os, datetime
from pathlib import Path
from visdom import Visdom
viz = Visdom(port=8850)
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset, ChexpertDataset
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# Saving images
from PIL import Image

from torchinfo import summary

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def main():
    args = create_argparser().parse_args()
    args.result_dir = f'{args.result_dir}_{datetime.date.today().strftime("%Y_%m_%d")}'

    dist_util.setup_dist()
    logger.configure(dir=args.result_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if args.dataset=='brats':
        ds = BRATSDataset(args.data_dir, test_flag=True)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False)

    elif args.dataset=='chexpert':
        # ds = ChexpertDataset(args.data_dir, class_cond=args.class_cond, test_flag=True)
        # data = th.utils.data.DataLoader(
        #     ds,
        #     batch_size=args.batch_size,
        #     shuffle=False)
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            deterministic=True
        )
        datal = iter(data)
   
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path)
    )
    print('loaded classifier')

    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print('pmodel', p1, 'pclass', p2)
    logger.log('pmodel', p1, 'pclass', p2)

    summary(model, input_data={
        'x': th.randn(size=(args.batch_size, 1, 256, 256), device=dist_util.dev()),
        'timesteps': diffusion._scale_timesteps(th.randint(0, 1, (args.batch_size,), device=dist_util.dev()).long()),
        'y':  th.zeros(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int),
        'p_uncond': -1,
        'clf_free': False
    })

    summary(classifier, input_data={
        'x': th.randn(size=(args.batch_size, 1, 256, 256), device=dist_util.dev()),
        'timesteps': diffusion._scale_timesteps(th.randint(0, 1, (args.batch_size,), device=dist_util.dev()).long()),
    })

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.long().view(-1)]
            a=th.autograd.grad(selected.sum(), x_in)[0]
            return a, a * args.classifier_scale

    def model_fn(x, t, y=None, p_uncond=-1, null=False, clf_free=False):
        assert y is not None
        return model(x, t, y if args.class_cond else None, p_uncond=-1, null=False, clf_free=False)

    logger.log("sampling...")
    all_orgs       = []
    all_images     = []
    all_org_labels = []
    all_tgt_labels = []
    all_names      = []
    all_results    = []

    # for img in datal:
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        img = next(datal)
        # img = next(data)  # should return an image from the dataloader "data"
        print('img', img[0].shape, img[1])
        if args.dataset=='brats':
            Labelmask = th.where(img[3] > 0, 1, 0)
            number=img[4][0]
            if img[2]==0:
                continue    #take only diseased images as input
                
            viz.image(visualize(img[0][0, 0, ...]), opts=dict(caption="img input 0"))
            viz.image(visualize(img[0][0, 1, ...]), opts=dict(caption="img input 1"))
            viz.image(visualize(img[0][0, 2, ...]), opts=dict(caption="img input 2"))
            viz.image(visualize(img[0][0, 3, ...]), opts=dict(caption="img input 3"))
            viz.image(visualize(img[3][0, ...]), opts=dict(caption="ground truth"))
        else:
            #   if th.equal(img[1]["y"], th.tensor([1])): 0 is diseased for chexpert
            #       continue # take only diseased images as input
            number = img[1]["name"]
            org_label = img[1]["y"].to(dist_util.dev())
            
            viz.image(visualize(img[0][0, ...]), opts=dict(caption=f"img {'diseased' if img[1]['y'] else 'healthy'} {number[0]}"))
            print('img1', img[1])
            print('number', number)

        if args.class_cond:
            classes = th.zeros(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int)
            model_kwargs["y"] = classes
            print('target y', model_kwargs["y"])
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        print('samplefn', sample_fn)
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()
        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4 if args.dataset == 'brats' else 1, args.image_size, args.image_size), img, org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level
        )
        end.record()
        th.cuda.synchronize()
        th.cuda.current_stream().synchronize()


        print('time for 1000', start.elapsed_time(end))

        if args.dataset=='brats':
            viz.image(visualize(sample[0,0, ...]), opts=dict(caption="sampled output0"))
            viz.image(visualize(sample[0,1, ...]), opts=dict(caption="sampled output1"))
            viz.image(visualize(sample[0,2, ...]), opts=dict(caption="sampled output2"))
            viz.image(visualize(sample[0,3, ...]), opts=dict(caption="sampled output3"))
            difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
            viz.heatmap(visualize(difftot), opts=dict(caption="difftot"))
          
        elif args.dataset=='chexpert':
            viz.image(visualize(sample[0, ...]), opts=dict(caption=f'sampled output {img[1]["name"][0]}'))
            diff=abs(visualize(org[0, 0,...])-visualize(sample[0, 0, ...]))
            diff=np.array(diff.cpu())
            # viz.heatmap(visualize(np.flipud(diff)), opts=dict(caption=f'diff {img[1]["name"][0]}'))
            cm = plt.get_cmap('jet') # Returns (R, G, B, A) in float64
            colored_diff = cm(visualize(diff))[:, :, :3]
            viz.image(colored_diff.transpose(2, 0, 1), opts=dict(caption=f'diff {img[1]["name"][0]}'))

            # Save image
            original_img = (np.concatenate((np.array(visualize(org[0, ...]).cpu()).transpose(1, 2, 0),) * 3, axis=-1) * 255).astype(np.uint8)
            sampled_img = (np.concatenate((np.array(visualize(sample[0, ...]).cpu()).transpose(1, 2, 0),) * 3, axis=-1) * 255).astype(np.uint8)
            heatmap_img = (colored_diff * 255).astype(np.uint8)

            result = np.hstack([original_img,
                                sampled_img,
                                heatmap_img])
            all_results.append(result)
            all_names.append(img[1]["name"][0])
            # End of save image

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_orgs = [th.zeros_like(org) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_orgs, org)  # gather not supported with NCCL
        all_orgs.extend([org.cpu().numpy() for org in gathered_orgs])

        if args.class_cond:
            gathered_org_labels = [
                th.zeros_like(org_label) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_org_labels, org_label)
            all_org_labels.extend([labels.cpu().numpy() for labels in gathered_org_labels])

            gathered_tgt_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_tgt_labels, classes)
            all_tgt_labels.extend([labels.cpu().numpy() for labels in gathered_tgt_labels])

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    org_arr = np.concatenate(all_orgs, axis=0)
    org_arr = org_arr[: args.num_samples]

    if args.class_cond:
        org_label_arr = np.concatenate(all_org_labels, axis=0)
        org_label_arr = org_label_arr[: args.num_samples]

        tgt_label_arr = np.concatenate(all_tgt_labels, axis=0)
        tgt_label_arr = tgt_label_arr[: args.num_samples]
        
        name_arr = np.array(all_names)
        name_arr = name_arr[: args.num_samples]
    
    if dist.get_rank() == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        shape_str = "x".join([str(x) for x in arr.shape])
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        out_path = os.path.join(args.result_dir, f"samples_{os.path.splitext(os.path.basename(args.model_path))[0]}_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, samples=arr, org_labels=org_label_arr, tgt_labels=tgt_label_arr, names=name_arr, orgs=org_arr)
        
        final_samples_image = Image.fromarray(np.vstack(all_results))
        logger.log(f"saving generated sample images to {args.result_dir}")
        final_samples_image.save(os.path.join(args.result_dir, f'latest_run_{os.path.splitext(os.path.basename(args.model_path))[0]}.png'))

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats',
        result_dir='results/',
        clf_free=False,
        guidance_scale=-1,
        p_uncond=-1,
        null=False
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

