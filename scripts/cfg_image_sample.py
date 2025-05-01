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
        # ds = ChexpertDataset(args.data_dir, class_cond=True, test_flag=True)
        # datal = th.utils.data.DataLoader(
        #     ds,
        #     batch_size=args.batch_size,
        #     shuffle=True)
        # datal = iter(datal)
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

    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    print('pmodel', p1)
    logger.log('pmodel', p1)

    def model_fn(x, t, y=None, p_uncond=-1, null=False, clf_free=True):
        assert y is not None
        return model(x, t, y if args.class_cond else None, p_uncond, null, clf_free)

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
            number=img[1]["name"]
            org_label = img[1]["y"].to(dist_util.dev())
            
            viz.image(visualize(img[0][0, ...]), opts=dict(caption=f"img {'healthy' if img[1]['y'] else 'diseased'} {number[0]}"))
            print('img1', img[1])
            print('number', number)
            print('org y', img[1]["y"])

        if args.class_cond:
            classes = th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int)
            model_kwargs = dict(
                y=classes,
                p_uncond=-1,
                clf_free=True
            )
            print('tgt y', model_kwargs["y"])
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
            cond_fn=None,
            device=dist_util.dev(),
            noise_level=args.noise_level,
            guidance_scale=args.guidance_scale
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
            diff=abs(org[0, 0,...] - sample[0,0, ...])
            diff=np.array(diff.cpu())
            # viz.heatmap(visualize(np.flipud(diff)), opts=dict(caption=f'diff {img[1]["name"][0]}'))
            cm = plt.get_cmap('jet') # Returns (R, G, B, A) in float64
            colored_diff = cm(visualize(diff))[:, :, :3]
            viz.image(colored_diff.transpose(2, 0, 1), opts=dict(caption=f'diff {img[1]["name"][0]}'))

            # Save image
            original_img = (np.concatenate((np.array(visualize(org[0, ...]).cpu()).transpose(1, 2, 0),) * 3, axis=-1) * 255).astype(np.uint8)
            sampled_img = (np.concatenate((np.array(visualize(sample[0, ...]).cpu()).transpose(1, 2, 0),) * 3, axis=-1) * 255).astype(np.uint8)
            heatmap_img = (colored_diff * 255).astype(np.uint8)

            # thresh = threshold_otsu(visualize(diff))
            # logger.log(f'threshold: {thresh}')
            # mask = th.where(th.tensor(visualize(diff)) > 0.5, 1, 0)  #this is our predicted binary segmentation
            # viz.image(visualize(mask), opts=dict(caption=f'mask {img[1]["name"][0]}'))

            # Convert mask to boxes
            # obj_ids = th.unique(mask)
            # obj_ids = obj_ids[1:]
            # masks = mask == obj_ids[:, None, None]

            # boxes = masks_to_boxes(masks)
            # drawn_boxes = draw_bounding_boxes((img[0][0, ...] * 255).type(th.uint8).repeat(3, 1, 1), boxes, colors="red")
            # fig, _ = show(drawn_boxes)
            # viz.image(drawn_boxes, opts=dict(caption=f'bbox {img[1]["name"][0]}'))

            # Image.fromarray((np.array(visualize(mask).cpu()) * 255).astype(np.uint8)).save(f'results/{args.data_dir.split("/")[-1]}/mask_{img[1]["name"][0]}.png')

            # Image.fromarray(heatmap_img).save(f'results/heatmap_{img[1]["name"][0]}.png')
            # Image.fromarray(sampled_img).save(f'results/sampled_{img[1]["name"][0]}.png')

            # mask = (np.array(mask.cpu().repeat(3, 1, 1)).transpose(1, 2, 0) * 255).astype(np.uint8)
            # drawn_boxes = np.array(drawn_boxes.cpu()).transpose(1, 2, 0).astype(np.uint8)

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
        noise_level=500,
        dataset='brats',
        num_classes=2,
        result_dir='./results',
        guidance_scale=2.0,
        class_cond=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

