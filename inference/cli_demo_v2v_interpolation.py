import os
import sys
import argparse
import torch
from PIL import Image
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel, CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
sys.path.insert(0, "./")
sys.path.insert(0, "../")
from pipelines.pipeline_cogvideox_image2video_mvaa import MVAACogVideoXImageToVideoPipeline
from utils.util import load_music_beats_to_video_motion

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def main(args):
    ####
    os.makedirs(args.output_video_path, exist_ok=True)
    ##### check whether the video already exists
    save_path = os.path.join(args.output_video_path, "output.mp4")
    if os.path.exists(save_path):
        print(f"Video already exists: {save_path}")
        return

    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
    else:
        generator = None

    # Load pretrained components
    transformer = CogVideoXTransformer3DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    text_encoder = T5EncoderModel.from_pretrained(args.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.bfloat16)

    pipe = MVAACogVideoXImageToVideoPipeline.from_pretrained(
        args.model_id, text_encoder=text_encoder, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16
    ).to("cuda")

    if args.lora_path:
        pipe.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=args.lora_scale)
        print(f"Successfully loaded LoRA from: {args.lora_path}")

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Load images
    frames_dir = args.video_path.replace(".mp4", "")
    if not os.path.exists(os.path.join(frames_dir, "frame_0000.png")):
        os.system(f"python preprocess/video_tools.py \
                        --input_path {args.video_path} \
                        --output_dir {frames_dir} \
                        --type 'mp4_to_images' ")
    images = [load_image(os.path.join(frames_dir, img))
            for img in sorted(os.listdir(frames_dir), key=lambda x: int(x.split('.')[0].split('_')[1]))] #.convert("RGB")
    print("Successfully loaded image(s)")
    
    # Load motion-beats alignment
    if args.music_beats_to_video_motion_path is not None and os.path.exists(args.music_beats_to_video_motion_path):
        music_beats_to_video_motion_path = args.music_beats_to_video_motion_path
        print(f"Extract motion-beats alignment from: {music_beats_to_video_motion_path}")
    else:
        print("Motion-beats alignment file is not exists. Extract now... ")
        if args.music_beats_to_video_motion_path is not None:
            music_beats_to_video_motion_path = args.music_beats_to_video_motion_path
        else:
            music_beats_to_video_motion_path = os.path.join(args.video_path.replace("videos", "music_beats_to_video_motion").replace(".mp4", ""), os.path.basename(args.music_path).replace(".mp3", ".csv"))
        if not os.path.exists(music_beats_to_video_motion_path):
            os.makedirs(os.path.dirname(music_beats_to_video_motion_path), exist_ok=True)
            
        os.system(f"python preprocess/motion_music_alignment.py \
                        --video {args.video_path} \
                        --audio {args.music_path} \
                        --output {music_beats_to_video_motion_path}")
    
    select_image_index, select_image_insert_index = load_music_beats_to_video_motion(music_beats_to_video_motion_path) 
    print(f"Select frame index: {select_image_index}")
    print(f"Select frame should be inserted index: {select_image_insert_index}")

    # Generate
    with torch.no_grad():
        result = pipe(
            image=images,
            prompt=args.prompt,
            guidance_scale=6,
            use_dynamic_cfg=True,
            num_inference_steps=50,
            num_frames=args.num_frames,
            select_image_index=select_image_index,
            select_image_insert_index=select_image_insert_index,
            generator=generator,
        ).frames[0]

    ##### save video to output_video_path
    export_to_video(result, save_path, fps=args.fps)
    print("Video saved to:", save_path)
    
    ### merge video and audio
    if args.music_path is not None:
        save_path_with_audio = save_path.replace(".mp4", "_withaudio.mp4")
        os.system(f"python postprocess/merge_videos_with_audio.py \
                        --videos {save_path} \
                        --audios {args.music_path} \
                        -o {save_path_with_audio} \
                        --start_time 0 \
                        --end_time 3 \
                ")
        print(f"Video with audio {args.music_path} saved to:", save_path_with_audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_video_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_scale", type=float, default=1.0 / 128)
    parser.add_argument("--music_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--music_beats_to_video_motion_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    args = parser.parse_args()
    
    main(args)
