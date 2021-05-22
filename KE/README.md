# Kinetic Energy at Landing
Both because NSL requires it, and so we can insure our stuff won't break, we will typically
need to calculate the kinetic energy of each independing of section of the rocket as it lands. 

The `terminalV` function will give a fairly accurate estimate of the terminal velocity of an object based on simple 
quadratic drag, proving you know the drag coefficient (we typically ignore the drag of the falling rocket and just use the drag of the parachute.

The `KEcalc` function will calculate the kinetic energy at landing in units of ft-lb since that's what NASA uses.
