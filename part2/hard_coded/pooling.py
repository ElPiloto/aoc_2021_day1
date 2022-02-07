import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

x = np.loadtxt('./aoc_input.txt')
kernel = jnp.array([1.0, 1.0, 1.0])

summed = jnp.convolve(x, kernel, mode='valid')
# subtract each number in sequence from preceding value
diff = jnp.diff(summed)
greater = diff > 0
solution = jnp.sum(greater)
print(f'Solution to part 2 is {solution}')
