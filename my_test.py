import jax.numpy as jnp

a=jnp.array([1,2,3])
b=jnp.array([2,3,4])
ego_pos = jnp.vstack((a, 
                    b))
missile_pos = jnp.vstack((a, 
                        b, 
                        a))

jnp.hstack()
print(ego_pos)
print(missile_pos)

c=ego_pos-missile_pos

print(c)