#activate python vitual env, optional  
# source /root/anaconda3/bin/activate
#experiment dir
exp_dir=/data/guixing/Lakers/
. $exp_dir/path.sh
export OMP_NUM_THREADS=3
set -e
export PYTHONPATH=$PYTHONPATH:$LAKER_ROOT

#training data dir must contain wav.scp and label.txt files
#wav.scp: standard kaldi wav.scp file, see https://kaldi-asr.org/doc/data_prep.html 
#label.txt: label text file, the format is, uttid sequence-of-integer, where integer
#           is one-based indexing mapped label, note that zero is reserved for blank,  
#           ,eg., utt_id_1 3 5 7 10 23 
train_data_dir="./bull/"




#available cuda devices and number of nodes
cuda_devices="4,5,6,7"
nnodes=1

batch_size=4
proto=cnn_deeptransformer

mkdir -p $exp_dir $exp_dir/.tmp
tmpdir=$exp_dir/.tmp

case 0 in
1)
;;
esac

CUDA_VISIBLE_DEVICES=$cuda_devices NCCL_DEBUG=TRACE python $LAKER_ROOT/trainer/train_cnn_otfaug.py \
  --data_dir $train_data_dir \
  --initial_lr 0.0003 \
  --final_lr 0.00001 \
  --grad_clip 3.0 \
  --num_batches_per_epoch 15012 \
  --num_epochs 15 \
  --momentum 0.9 \
  --cuda --batch_size 8 \
  --dropout 0.3 \
  --stride 2 \
  --loader otf_imag \
  --seq_len 20 \
  --local_rank 0 \
  --input_H 640 \
  --input_W 480 \
  --cnn_layers 5 \
  --transformer_layers 5 \
  $proto \
  $exp_dir/logs/train_deepcnn_deeptranormer.log \
  $exp_dir/output2/ > $exp_dir/logs/main.log

exit 0
