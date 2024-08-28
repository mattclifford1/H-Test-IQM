## noise_autoencoders
px models are trained on the distribution of natural images
ux models are on uniform dist (noise)


## save_nets
no letter = natural images training
u = uniform trained
number is the centers
mae doesn't work -- ignore those weights


# embendding
8x8 -- flatten this (get rid of spacial aspect)
64 embedding
counts per value per embedding (of the coding)

coding (code vectors)
2 centers: -1, 1            
5 centers: -2, -1, 0, 1, 2


# best looking nets
mse-2/5 (px)   -   best
nlpd-2/5-u     -   interesting


# using the nets
plot distribution of some features


# cifar
resize the images to correct input size


# difference in features values
compare all feautres then get mean difference


# how to get pdf/pmf for comparison
think about this...