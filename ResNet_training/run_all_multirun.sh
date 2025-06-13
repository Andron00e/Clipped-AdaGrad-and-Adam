

if [[ " $@ " =~ " 0 " ]]; then
    echo "run sgd-layerwise-miltirun"
    python3 multi_runs_cifar10.py --config-name resnet_parent_multirun \
        opt.clipping="layerwise" \
        opt.max_grad_norm=5.0 \
        global_.wandb_run_name="sgd-layerwise-multirun" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run adam-layerwise-miltirun"
    python3 multi_runs_cifar10.py --config-name resnet_parent_multirun \
        opt.clipping="layerwise" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="adam-layerwise-miltirun" \
        opt.optimizer=adam
fi

if [[ " $@ " =~ " 1 " ]]; then
    echo "run sgd-elementwise-multirun"
    python3 multi_runs_cifar10.py --config-name resnet_parent_multirun \
        opt.clipping="elementwise" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="sgd-elementwise-multirun" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run adam-elementwise-multirun"
    python3 multi_runs_cifar10.py --config-name resnet_parent_multirun \
        opt.clipping="elementwise" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="adam-elementwise-multirun" \
        opt.optimizer=adam
fi

if [[ " $@ " =~ " 2 " ]]; then
    echo "run sgd-global-multirun"
    python3 multi_runs_cifar10.py --config-name resnet_parent_multirun \
        opt.clipping="global" \
        opt.max_grad_norm=5.0 \
        global_.wandb_run_name="sgd-global-multirun" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run adam-global-multirun"
    python3 multi_runs_cifar10.py --config-name resnet_parent_multirun \
        opt.clipping="global" \
        opt.max_grad_norm=1.0 \
        global_.wandb_run_name="adam-global-multirun" \
        opt.optimizer=adam
fi

if [[ " $@ " =~ " 3 " ]]; then
    echo "run sgd-local-multirun"
    python3 multi_runs_cifar10.py --config-name resnet_parent_multirun \
        opt.clipping="local" \
        opt.max_grad_norm=2.5 \
        global_.wandb_run_name="sgd-local-multirun" \
        opt.optimizer=sgd opt.lr=1e-2

    echo "run adam-local-multirun"
    python3 multi_runs_cifar10.py --config-name resnet_parent_multirun \
        opt.clipping="local" \
        opt.max_grad_norm=0.1 \
        global_.wandb_run_name="adam-local-multirun" \
        opt.optimizer=adam
fi
