#!/bin/bash
. path.sh
. cmd.sh
set -ex
set -o pipefail
my_nj=28
stage=10
current_dir=$(pwd)
task=$current_dir/NOW_Online
lists=$task/lists
conf=$task/conf
exp=$task/exp
corpus=$task/corpus
data=$task/data
local=$task/local
mkdir -p $task $conf $lists $local $exp $corpus $data
####################################################
###           generate corpus                   ###
###################################################
#stage=0
if [ $stage -le 0 ]; then
    ######## we should have .text and .scp data
    srcdir="wiki_NowOnline_20181005_20181115"   ### src data dir
    #array=("20181005_20181025" "20181026_20181101" "20181109_20181115" "20181116_20181220" "20181210_20181212" "20181213_20181220" "20181221_20181224" "20181224_20181226")
    array=("20181226_20190109")

    for x in ${array[@]};do
        echo "prreprocess data in $x"
	# generate wav from url
        #[ ! -f $srcdir/${x}.scp ] && \
        #    python $local/generate_wavscp_text.py $srcdir/${x}.result $srcdir/${x}.scp $srcdir/${x}.text

        for files in scp text;do
            [ ! -f $lists/${x}.$files ] && \
                cp $srcdir/${x}.$files $lists/${x}.$files
        done

        dir=$corpus/$x && mkdir -p $dir
        [ ! -f $dir/generate_wav.sh ] && \
            max_dur=10 && min_dur=0.20 && \
            cat $lists/${x}.scp | awk 'NF==18 && $17>min_dur && $17 < max_dur {print $2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "$10" "$11" "$12" "$13" "dir "/" $1".wav "$15" "$16" "$17}' min_dur=$min_dur max_dur=$max_dur dir=$dir > $dir/generate_wav.sh && \
            chmod 755 $dir/generate_wav.sh && bash $dir/generate_wav.sh

        data_dir=$data/$x && mkdir -p $data_dir
        [ ! -f $data_dir/sph.flist ] && \
            find $corpus/$x -iname "*.wav" | sort > $data_dir/sph.flist && \
            sed -e 's?.*/??' -e 's?.wav??' $data_dir/sph.flist | paste - $data_dir/sph.flist | sort > $data_dir/wav.scp

        [ ! -f $data_dir/utt2spk ] && \
            awk '{print $1 " " $1}' $data_dir/wav.scp > $data_dir/utt2spk
        [ ! -f $data_dir/spk2utt ] && \
            utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt && \
            cat $data_dir/utt2spk | awk '{print $1}' > $data_dir/utts && \
            utils/filter_scp.pl $data_dir/utts $lists/${x}.text > $data_dir/text
        #continue
        dataset_dir=$exp/$x
        dataset=$(basename $dataset_dir)
        if [ ! -d ${dataset_dir}_hires ]; then
            utils/copy_data_dir.sh $data/$x ${dataset_dir}_hires
        fi

        mfccdir=$exp/make_mfcc_hires/$dataset/mfccdir
        if [ ! -f ${dataset_dir}_hires/.mfcc.done ]; then
            steps/make_mfcc_pitch_online.sh --nj $my_nj --mfcc-config $conf/mfcc_hires.conf \
                --cmd "$train_cmd" ${dataset_dir}_hires $exp/make_mfcc_hires/$dataset $mfccdir;
            steps/compute_cmvn_stats.sh ${dataset_dir}_hires $exp/make_mfcc_hires/${dataset} $mfccdir;
            utils/fix_data_dir.sh ${dataset_dir}_hires;
            touch ${dataset_dir}_hires/.mfcc.done
            touch ${dataset_dir}_hires/.done
        fi

        data_dir=${dataset_dir}_hires
        [ ! -f $data_dir/feats.npz ] && \
            apply-cmvn --norm-means --norm-vars --utt2spk=ark:$data_dir/utt2spk scp:$data_dir/cmvn.scp scp:$data_dir/feats.scp ark,t:$data_dir/feats.txt && \
            python $local/write_kaldi_npz.py $data_dir/feats.txt $data_dir/feats.npz && rm -rf $data_dir/feats.txt
    done
    ############extracted mfcc features and converted to npz format
    exit 0
fi
#################################################
###         generate train/dev set            ###
################################################
#stage=1
if [ $stage -le 1 ]; then
    #array=("20181005_20181025" "20181026_20181101" "20181109_20181115" "20181116_20181220" "20181210_20181212" "20181213_20181220" "20181221_20181224" "20181224_20181226")
    array=("20181226_20190109")
    #for x in ${array[@]};do
    #    [ ! -f $exp/${x}_hires/feats_new.npz ] && \
    #        python $local/convert_wordids.py $lists/keywords.txt $exp/${x}_hires/feats.npz $exp/${x}_hires/text $exp/${x}_hires/feats_new.npz
    #done
    #exit 0
    [ ! -f $lists/keywords_42.txt ] && \
        python $local/generate_keywordslist.py $exp/20181221_20181224_hires/text $lists/keywords.txt $lists/keywords_42.txt

    dir=$exp/embeddings_20190108 && rm -rf $dir && mkdir -p $dir
    [ ! -f $dir/train.npz ] && \
        python $local/generate_training_sets.py $lists/keywords_42.txt $exp $dir/train.npz #$dir/dev.npz
    [ ! -f $dir/dev.npz ] && \
        ln -s $exp/20181221_20181224_hires/feats_new1.npz $dir/dev.npz
    exit 0
fi

stage=2
if [ $stage -le 2 ]; then
    data_dir=$exp/embeddings_20190107_1
    #data_dir=$exp/embeddings_20181221
    hs=256; m=0.15;  lr=0.0001;   kp=0.6; bs=400; epo=499;   init=0.05
    sets=1; n_same_paris=10000; output_size=0
    lastepoch=454;
    q_lamda=0.0; distance_metric="cosine"; gpu_device=$1
    model_dir=$data_dir/Saved_Models1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_ori
    output_dir=$data_dir/Outputs1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_ori
    #model_dir=$data_dir/../embeddings_20181221/Saved_Models1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_neg2
    #output_dir=$data_dir/../embeddings_20181221/Outputs1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_neg2
    mkdir -p $model_dir $output_dir
    name=${hs}_${m}_${lr}_${kp}_${bs}_${epo}_${init}_set${sets}_${n_same_paris}_${output_size}
    #rm -rf $output_dir/${name}.txt $model_dir/${name}*
    #python $local/awe_individual/2biLSTM_mtl_bnf_lesspairs_new.py $data_dir $model_dir $output_dir -hs ${hs} -m ${m} -lr ${lr} -kp ${kp} -bs ${bs} -epo ${epo} -init ${init} -sets ${sets} -n_same_pairs ${n_same_paris} -output_size ${output_size} -gpu_device ${gpu_device} #-lastepoch ${lastepoch}
    #exit 0

    python $local/awe_individual/apply_embedding.py $data_dir $model_dir $output_dir -hs ${hs} -m ${m} -lr ${lr} -kp ${kp} -bs ${bs} -epo ${epo} -init ${init} -sets ${sets} -n_same_pairs ${n_same_paris} -output_size ${output_size} -gpu_device ${gpu_device} -lastepoch ${lastepoch}

    #python $local/analysis_log_thr.py $output_dir/${name}.txt $output_dir/${name}.png model16
    echo "sucess"
    #cd /data/home/yougenyuan/software/simpleMultiGPU
    #CUDA_VISIBLE_DEVICES=$gpu_device ./trainKaldiChainModel #./simpleMultiGPU
    #paste wiki_NowOnline_20181005_20181115/20181221_20181224.result_new 20181221_20181224.result3_-0.08 > 20181221_20181224.result3_new
    #cut -d" " -f5,7,10 20181221_20181224.result3_new > 20181221_20181224.txt
    #paste 20181221_20181224.txt 20181221_20181224.result4_0.61 &> 20181221_20181224.result4_new
    #cut -d" " -f2,6 20181221_20181224.result4_new > 20181221_20181224_new.txt
    #paste 20181221_20181224.txt 20181221_20181224.result6_0.0 &> 20181221_20181224.result6_new
    #cut -d" " -f2,6 20181221_20181224.result6_new > 20181221_20181224_new2.txt
    #exit 0
fi
#echo "sucess"
exit 0

#stage=2
#if [ $stage -le 2 ]; then
#    data_dir=$exp/embeddings_20190107_1
#    #data_dir=$exp/embeddings_20181221
#    hs=256; m=0.15;  lr=0.0001;   kp=0.6; bs=400; epo=499;   init=0.05
#    sets=1; n_same_paris=10000; output_size=0
#    lastepoch=-1 #454;
#    q_lamda=0.0; distance_metric="cosine"; gpu_device=$1
#    model_dir=$data_dir/Saved_Models1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_13
#    output_dir=$data_dir/Outputs1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_13
#    #model_dir=$data_dir/../embeddings_20181221/Saved_Models1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_neg2
#    #output_dir=$data_dir/../embeddings_20181221/Outputs1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_neg2
#    mkdir -p $model_dir $output_dir
#    name=${hs}_${m}_${lr}_${kp}_${bs}_${epo}_${init}_set${sets}_${n_same_paris}_${output_size}
#    #rm -rf $output_dir/${name}.txt $model_dir/${name}*
#    #python $local/awe_individual/2biLSTM_mtl_bnf_lesspairs_new.py $data_dir $model_dir $output_dir -hs ${hs} -m ${m} -lr ${lr} -kp ${kp} -bs ${bs} -epo ${epo} -init ${init} -sets ${sets} -n_same_pairs ${n_same_paris} -output_size ${output_size} -gpu_device ${gpu_device} #-lastepoch ${lastepoch}
#    cd /data/home/yougenyuan/software/simpleMultiGPU
#    CUDA_VISIBLE_DEVICES=$gpu_device ./trainKaldiChainModel #./simpleMultiGPU
#    exit 0
#
#    python $local/awe_individual/apply_embedding.py $data_dir $model_dir $output_dir -hs ${hs} -m ${m} -lr ${lr} -kp ${kp} -bs ${bs} -epo ${epo} -init ${init} -sets ${sets} -n_same_pairs ${n_same_paris} -output_size ${output_size} -gpu_device ${gpu_device} -lastepoch ${lastepoch}
#
#    #python $local/analysis_log_thr.py $output_dir/${name}.txt $output_dir/${name}.png model16
#    echo "sucess"
#    #cd /data/home/yougenyuan/software/simpleMultiGPU
#    #CUDA_VISIBLE_DEVICES=$gpu_device ./trainKaldiChainModel #./simpleMultiGPU
#    #paste wiki_NowOnline_20181005_20181115/20181221_20181224.result_new 20181221_20181224.result3_-0.08 > 20181221_20181224.result3_new
#    #cut -d" " -f5,7,10 20181221_20181224.result3_new > 20181221_20181224.txt
#    #paste 20181221_20181224.txt 20181221_20181224.result4_0.61 &> 20181221_20181224.result4_new
#    #cut -d" " -f2,6 20181221_20181224.result4_new > 20181221_20181224_new.txt
#    paste 20181221_20181224.txt 20181221_20181224.result5_0.0 &> 20181221_20181224.result5_new
#    cut -d" " -f2,6 20181221_20181224.result5_new > 20181221_20181224_new1.txt
#    exit 0
#fi
#echo "sucess"


#stage=2
#if [ $stage -le 3 ]; then
#    ###### adding attention on keywords verification
#    ### but the experiments show that it has no improvement
#    data_dir=$exp/embeddings_20190107_1
#    hs=256; m=0.150;  lr=0.0001;   kp=0.6; bs=400; epo=499;   init=0.05
#    sets=1; n_same_paris=10000; output_size=0
#    lastepoch=0; q_lamda=0.0; distance_metric="cosine"; gpu_device=$1
#    model_dir=$data_dir/Saved_Models1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_ori
#    output_dir=$data_dir/Outputs1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_att4
#    mkdir -p $model_dir $output_dir
#    name=${hs}_${m}_${lr}_${kp}_${bs}_${epo}_${init}_set${sets}_${n_same_paris}_${output_size}
#    #rm -rf $output_dir/${name}.txt $model_dir/${name}*
#    python $local/awe_individual/2biLSTM_mtl_bnf_attentions.py $data_dir $model_dir $output_dir -hs ${hs} -m ${m} -lr ${lr} -kp ${kp} -bs ${bs} -epo ${epo} -init ${init} -sets ${sets} -n_same_pairs ${n_same_paris} -output_size ${output_size} -gpu_device ${gpu_device} #-lastepoch ${lastepoch}
#    cd /data/home/yougenyuan/software/simpleMultiGPU
#    CUDA_VISIBLE_DEVICES=$gpu_device ./trainKaldiChainModel
#    exit 0
#
#    #python $local/awe_individual/apply_embedding.py $data_dir $model_dir $output_dir -hs ${hs} -m ${m} -lr ${lr} -kp ${kp} -bs ${bs} -epo ${epo} -init ${init} -sets ${sets} -n_same_pairs ${n_same_paris} -output_size ${output_size} -gpu_device ${gpu_device} -lastepoch ${lastepoch}
#
#    #python $local/analysis_log.py $output_dir/${name}.txt $output_dir/${name}.png model1
#    echo "sucess"
#    #cd /data1/home/yougenyuan/software/simpleMultiGPU
#    #./simpleMultiGPU
#fi

stage=3
if [ $stage -le 4 ]; then
    echo "model freezing ..."
    data_dir=$exp/embeddings_20190107_1
    hs=256; m=0.15;  lr=0.0001;   kp=0.6; bs=400; epo=499;   init=0.05
    sets=1; n_same_paris=10000; output_size=0
    lastepoch=454; q_lamda=0.0; distance_metric="cosine"; gpu_device=$1
    model_dir=$data_dir/Saved_Models1_mtl_bnf_contextpadding_fc1_${output_size}_${q_lamda}_set${sets}_${n_same_paris}_${output_size}_lesspairs_news4_ori

    name=${hs}_${m}_${lr}_${kp}_${bs}_${epo}_${init}_set${sets}_${n_same_paris}_${output_size}
    #cd /data1/home/yougenyuan/code/tensorflow-r1.4
    #bazel-bin/tensorflow/python/tools/freeze_graph
    [ ! -f $model_dir/${name}-${lastepoch}.pb ] && \
        python $local/awe_individual/freeze_graph.py --input_graph=$model_dir/${name}-${lastepoch}.pbtxt --input_binary=false --input_checkpoint=$model_dir/${name}-${lastepoch} --output_graph=$model_dir/${name}-${lastepoch}.pb  --output_node_names="model/div"
    ### using the freezed model to tensorflow C++ API
fi


