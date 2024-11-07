export HF_ENDPOINT=https://hf-mirror.com
output_dir="/workspace/models/DocLayout-YOLO"

# Create output directory if it doesn't exist
mkdir -p ${output_dir}

# Download all model checkpoints
# huggingface-cli download --resume-download vikp/surya_det3 --local-dir ${output_dir}/surya_det3
# huggingface-cli download --resume-download vikp/surya_rec2 --local-dir ${output_dir}/surya_rec2
# huggingface-cli download --resume-download vikp/surya_layout3 --local-dir ${output_dir}/surya_layout3
# huggingface-cli download --resume-download vikp/surya_order --local-dir ${output_dir}/surya_order
# huggingface-cli download --resume-download vikp/surya_tablerec --local-dir ${output_dir}/surya_tablerec

# huggingface-cli download --resume-download vikp/surya_layout4 --local-dir ${output_dir}/surya_layout4
huggingface-cli download --resume-download juliozhao/DocLayout-YOLO-DocSynth300K-pretrain --local-dir ${output_dir}/DocSynth300K-pretrain

huggingface-cli download --resume-download juliozhao/DocLayout-YOLO-D4LA-from_scratch --local-dir ${output_dir}/D4LA-from_scratch
huggingface-cli download --resume-download juliozhao/DocLayout-YOLO-D4LA-Docsynth300K_pretrained --local-dir ${output_dir}/D4LA-Docsynth300K_pretrained

huggingface-cli download --resume-download juliozhao/DocLayout-YOLO-DocLayNet-from_scratch --local-dir ${output_dir}/DocLayNet-from_scratch
huggingface-cli download --resume-download juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained --local-dir ${output_dir}/DocLayNet-Docsynth300K_pretrained

benchmark_dir="/workspace/datasets/DocLayout-YOLO"
mkdir -p ${benchmark_dir}
# Download benchmark datasets (optional, commented out by default)
# huggingface-cli download --repo-type dataset --resume-download vikp/doclaynet_bench --local-dir ${benchmark_dir}/doclaynet_bench
# huggingface-cli download --repo-type dataset --resume-download vikp/rec_bench --local-dir ${benchmark_dir}/rec_bench
# huggingface-cli download --repo-type dataset --resume-download vikp/publaynet_bench --local-dir ${benchmark_dir}/publaynet_bench
# huggingface-cli download --repo-type dataset --resume-download vikp/order_bench --local-dir ${benchmark_dir}/order_bench
# huggingface-cli download --repo-type dataset --resume-download vikp/fintabnet_bench --local-dir ${benchmark_dir}/fintabnet_bench

# huggingface-cli download --repo-type dataset --resume-download juliozhao/doclayout-yolo-D4LA --local-dir ${benchmark_dir}/doclayout-yolo-D4LA
# huggingface-cli download --repo-type dataset --resume-download juliozhao/doclayout-yolo-DocLayNet --local-dir ${benchmark_dir}/doclayout-yolo-DocLayNet

echo "Download completed!"
