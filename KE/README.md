# Kinetic Energy at Landing
Both because NSL requires it, and so we can insure our stuff
won't break, we will typically need to calculate the
kinetic energy of each independent section of the rocket as it lands. 

The `terminalV` function will give a fairly accurate estimate
of the terminal velocity of an object based on simple quadratic
drag, providing you know the drag coefficient. We typically ignore the 
drag of the falling rocket and just use the drag of the parachute,
which gives an inherant safety factor.

