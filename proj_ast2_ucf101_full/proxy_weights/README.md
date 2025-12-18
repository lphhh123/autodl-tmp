
# Proxy weights directory

请将每个 GPU 的三种 layerwise 代理模型权重文件放在此目录下，文件名约定为：

- `layer_proxy_ms_2080ti.pt`
- `layer_proxy_mem_2080ti.pt`
- `layer_proxy_power_dyn_2080ti.pt`

- `layer_proxy_ms_3090.pt`
- `layer_proxy_mem_3090.pt`
- `layer_proxy_power_dyn_3090.pt`

- `layer_proxy_ms_4090.pt`
- `layer_proxy_mem_4090.pt`
- `layer_proxy_power_dyn_4090.pt`

如果你有静态功耗版本，也可以另外加：

- `layer_proxy_power_2080ti.pt`
- `layer_proxy_power_3090.pt`
- `layer_proxy_power_4090.pt`

`LayerHwProxy` 会自动根据 `device_name` 选择相应的权重文件。
