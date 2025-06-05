

if [[ " $@ " =~ " 0 " ]]; then
    echo "run layerwise-2.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="layerwise" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="layerwise-2.5" \
        opt.optimizer=adam

    echo "run layerwise-1.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="layerwise" \
        opt.max_grad_norm=1.0 \
        global_.wandb_run_name="layerwise-1.0" \
        opt.optimizer=adam

    echo "run layerwise-0.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="layerwise" \
        opt.max_grad_norm=0.5 \
        global_.wandb_run_name="layerwise-0.5" \
        opt.optimizer=adam
    
    echo "run layerwise-0.1"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="layerwise" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="layerwise-0.1" \
        opt.optimizer=adam
fi

if [[ " $@ " =~ " 1 " ]]; then
    echo "run elementwise-2.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="elementwise" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="elementwise-2.5" \
        opt.optimizer=adam

    echo "run elementwise-1.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="elementwise" \
        opt.max_grad_norm=1.0 \
        global_.wandb_run_name="elementwise-1.0" \
        opt.optimizer=adam

    echo "run elementwise-0.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="elementwise" \
        opt.max_grad_norm=0.5 \
        global_.wandb_run_name="elementwise-0.5" \
        opt.optimizer=adam
    
    echo "run elementwise-0.1"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="elementwise" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="elementwise-0.1" \
        opt.optimizer=adam
fi

if [[ " $@ " =~ " 2 " ]]; then
    echo "run global-10.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=10.0 \
        global_.wandb_run_name="global-10.0" \
        opt.optimizer=adam

    echo "run global-5.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=5.0 \
        global_.wandb_run_name="global-5.0" \
        opt.optimizer=adam

    echo "run global-2.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="global-2.5" \
        opt.optimizer=adam

    echo "run global-1.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=1.0 \
        global_.wandb_run_name="global-1.0" \
        opt.optimizer=adam

    echo "run global-0.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=0.5 \
        global_.wandb_run_name="global-0.5" \
        opt.optimizer=adam
    
    echo "run global-0.1"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="global-0.1" \
        opt.optimizer=adam
fi

if [[ " $@ " =~ " 3 " ]]; then
    echo "run local-2.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="local" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="local-2.5" \
        opt.optimizer=adam

    echo "run local-1.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="local" \
        opt.max_grad_norm=1.0 \
        global_.wandb_run_name="local-1.0" \
        opt.optimizer=adam

    echo "run local-0.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="local" \
        opt.max_grad_norm=0.5 \
        global_.wandb_run_name="local-0.5" \
        opt.optimizer=adam
    
    echo "run local-0.1"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="local" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="local-0.1" \
        opt.optimizer=adam
fi
