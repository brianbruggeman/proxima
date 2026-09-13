#include <mlx/mlx.h>
#include <cstring>
#include <vector>

using mlx::core::array;
using mlx::core::Shape;
using mlx::core::float32;

static array borrowed(float* data, Shape shape) {
  return array(data, std::move(shape), float32, [](void*) {});
}

extern "C" int proxima_mlx_gdn_scan(
    const float* query,
    const float* key,
    const float* value,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    int positions,
    int key_dim,
    int value_dim,
    int heads,
    float inv_sqrt_key_dim) {
  if (positions <= 0 || key_dim <= 0 || value_dim <= 0 || heads <= 0) return 1;
  const int key_heads = key_dim * heads;
  const int value_heads = value_dim * heads;
  array state_array = borrowed(state, {key_dim, value_dim, heads});
  for (int position = 0; position < positions; ++position) {
    array query_array = borrowed(
        const_cast<float*>(query + position * key_heads), {key_dim, heads});
    array key_array = borrowed(
        const_cast<float*>(key + position * key_heads), {key_dim, heads});
    array value_array = borrowed(
        const_cast<float*>(value + position * value_heads), {value_dim, heads});
    array gate_array = borrowed(const_cast<float*>(gate + position * heads), {heads});
    array beta_array = borrowed(const_cast<float*>(beta + position * heads), {heads});
    array decay = mlx::core::reshape(mlx::core::exp(gate_array), {1, 1, heads});
    array predicted = mlx::core::sum(
        state_array * decay * mlx::core::reshape(key_array, {key_dim, 1, heads}), 0);
    array delta = (value_array - predicted) * mlx::core::reshape(beta_array, {1, heads});
    state_array = state_array * decay +
        mlx::core::reshape(key_array, {key_dim, 1, heads}) * mlx::core::reshape(delta, {1, value_dim, heads});
    array readout = mlx::core::sum(
        state_array * mlx::core::reshape(query_array, {key_dim, 1, heads}), 0) * inv_sqrt_key_dim;
    state_array.eval();
    readout.eval();
    std::memcpy(state, state_array.data<float>(),
                static_cast<size_t>(key_dim * value_dim * heads) * sizeof(float));
    std::memcpy(output + position * value_heads, readout.data<float>(),
                static_cast<size_t>(value_dim * heads) * sizeof(float));
  }
  return 0;
}

extern "C" const char* proxima_mlx_bridge_name() {
  return "mlx";
}
