import happi
S = happi.Open(".")
S.ParticleBinning(0, average={"z":"all"}).animate(movie='visualization/anim.mp4')