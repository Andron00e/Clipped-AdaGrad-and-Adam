
CKPTS=("none" "epoch=15-step=1568.ckpt" "epoch=31-step=3136.ckpt" "epoch=47-step=4704.ckpt")

if [[ " $@ " =~ " 0 " ]]; then
    echo "run sgd-layerwise"
    for ckpt in "${CKPTS[@]}"; do
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=5.0 \
            global_.wandb_run_name="sgd-layerwise-5.0" \
            opt.optimizer=sgd opt.lr=1e-2 \
            global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/sgd-layerwise-5.0/${esc_ckpt} \
            opt.do_training=False
    done

    echo "run adam-layerwise"
    for ckpt in "${CKPTS[@]}"; do    
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=2.5 \
            global_.wandb_run_name="layerwise-2.5" \
            opt.optimizer=adam \
            "global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/layerwise-2.5/${esc_ckpt}" \
            opt.do_training=False
    done

fi

if [[ " $@ " =~ " 1 " ]]; then
    echo "run sgd-elementwise"
    for ckpt in "${CKPTS[@]}"; do
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=2.5 \
            global_.wandb_run_name="sgd-elementwise-2.5" \
            opt.optimizer=sgd opt.lr=1e-2 \
            "global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/sgd-elementwise-2.5/${esc_ckpt}" \
            opt.do_training=False
    done

    echo "run adam-elementwise"
    for ckpt in "${CKPTS[@]}"; do    
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=0.1 \
            global_.wandb_run_name="elementwise-0.1" \
            opt.optimizer=adam \
            "global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/elementwise-0.1/${esc_ckpt}" \
            opt.do_training=False
    done
fi

if [[ " $@ " =~ " 2 " ]]; then
    echo "run sgd-global"
    for ckpt in "${CKPTS[@]}"; do
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=5.0 \
            global_.wandb_run_name="sgd-global-5.0" \
            opt.optimizer=sgd opt.lr=1e-2 \
            "global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/sgd-global-5.0/${esc_ckpt}" \
            opt.do_training=False
    done

    echo "run adam-global"
    for ckpt in "${CKPTS[@]}"; do    
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=1.0 \
            global_.wandb_run_name="global-1.0" \
            opt.optimizer=adam \
            "global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/global-1.0/${esc_ckpt}" \
            opt.do_training=False
    done
fi

if [[ " $@ " =~ " 3 " ]]; then
    echo "run sgd-local"
    for ckpt in "${CKPTS[@]}"; do
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=2.5 \
            global_.wandb_run_name="sgd-local-2.5" \
            opt.optimizer=sgd opt.lr=1e-2 \
            "global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/sgd-local-2.5/${esc_ckpt}" \
            opt.do_training=False
    done

    echo "run adam-local"
    for ckpt in "${CKPTS[@]}"; do    
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=0.1 \
            global_.wandb_run_name="local-0.1" \
            opt.optimizer=adam \
            "global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/local-0.1/${esc_ckpt}" \
            opt.do_training=False
    done
fi

if [[ " $@ " =~ " 4 " ]]; then
    for ckpt in "${CKPTS[@]}"; do
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=10.0 \
            global_.wandb_run_name="sgd-none-10.0" \
            opt.optimizer=sgd opt.lr=1e-2 \
            "global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/sgd-none-10.0/${esc_ckpt}" \
            opt.do_training=False
    done

fi

if [[ " $@ " =~ " 5 " ]]; then
    for ckpt in "${CKPTS[@]}"; do    
        esc_ckpt=${ckpt//=/\\=}
        python3 check_tails_cifar10.py --config-name resnet_parent \
            opt.clipping="none" \
            opt.max_grad_norm=10.0 \
            global_.wandb_run_name="none-10.0" \
            opt.optimizer=adam \
            "global_.save_model_path=/scratch/izar/skorokho/cs439opt_scratch/none-10.0/${esc_ckpt}" \
            opt.do_training=False
    done
fi


