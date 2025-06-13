

if [[ " $@ " =~ " 0 " ]]; then
    echo "run sgd-layerwise-2.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="layerwise" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="sgd-layerwise-2.5" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-layerwise-1.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="layerwise" \
        opt.max_grad_norm=1.0 \
        global_.wandb_run_name="sgd-layerwise-1.0" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-layerwise-0.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="layerwise" \
        opt.max_grad_norm=0.5 \
        global_.wandb_run_name="sgd-layerwise-0.5" \
        opt.optimizer=sgd opt.lr=1e-2
    
    echo "run sgd-layerwise-0.1"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="layerwise" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="sgd-layerwise-0.1" \
        opt.optimizer=sgd opt.lr=1e-2
fi

if [[ " $@ " =~ " 1 " ]]; then
    echo "run sgd-elementwise-2.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="elementwise" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="sgd-elementwise-2.5" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-elementwise-1.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="elementwise" \
        opt.max_grad_norm=1.0 \
        global_.wandb_run_name="sgd-elementwise-1.0" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-elementwise-0.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="elementwise" \
        opt.max_grad_norm=0.5 \
        global_.wandb_run_name="sgd-elementwise-0.5" \
        opt.optimizer=sgd opt.lr=1e-2
    
    echo "run sgd-elementwise-0.1"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="elementwise" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="sgd-elementwise-0.1" \
        opt.optimizer=sgd opt.lr=1e-2
fi

if [[ " $@ " =~ " 2 " ]]; then
    echo "run sgd-global-10.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=10.0 \
        global_.wandb_run_name="sgd-global-10.0" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-global-5.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=5.0 \
        global_.wandb_run_name="sgd-global-5.0" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-global-2.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="sgd-global-2.5" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-global-1.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=1.0 \
        global_.wandb_run_name="sgd-global-1.0" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-global-0.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=0.5 \
        global_.wandb_run_name="sgd-global-0.5" \
        opt.optimizer=sgd opt.lr=1e-2
    
    echo "run sgd-global-0.1"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="global" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="sgd-global-0.1" \
        opt.optimizer=sgd opt.lr=1e-2
fi

if [[ " $@ " =~ " 3 " ]]; then
    echo "run sgd-local-2.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="local" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="sgd-local-2.5" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-local-1.0"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="local" \
        opt.max_grad_norm=1.0 \
        global_.wandb_run_name="sgd-local-1.0" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run sgd-local-0.5"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="local" \
        opt.max_grad_norm=0.5 \
        global_.wandb_run_name="sgd-local-0.5" \
        opt.optimizer=sgd opt.lr=1e-2
    
    echo "run sgd-local-0.1"
    python3 one_run_cifar10.py --config-name resnet_parent \
        opt.clipping="local" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="sgd-local-0.1" \
        opt.optimizer=sgd opt.lr=1e-2
fi

# python3 one_run_cifar10.py --config-name resnet_parent \
#     opt.clipping="none" \
#     opt.max_grad_norm=10.0 \
#     global_.wandb_run_name="sgd-none-10.0" \
#     opt.optimizer=sgd opt.lr=1e-2
