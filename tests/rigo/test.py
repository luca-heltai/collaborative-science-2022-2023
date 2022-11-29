import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')

print("Test code using numpy and jax with cpu.")

arr = np.arange(10)

arrnp = np.random.permutation(arr)
print(f"Permuted with numpy: {arrnp}.")

key = jax.random.PRNGKey(13)
arrjnp = jax.random.permutation(key, arr)
assert len(arrjnp) == 10, "Something is wrong with the jax installation."
print("Jax installation does not have issues")
print(f"Permuted with jax: {arrjnp}.")
