give each residue a chain index

some residues have no coordinates (esp in the middle of the chain): just ignore those residue
- this means you need to modify your nan-removing code / think of a new way to deal with that
write documentation

AAindex as extra features for AA embedding?

figure out why the net nans sometimes

600_rms_term1_edge+node-query (done)
600_rms_term1 (done)
600-2600_rms_term1 
600-2600_term1
600-2600_rms_term4 (done)
600-2600_term4
600_rms_term4_edge+node-query
600-2600_term4_edge+node-query

run more designs on the small complexes database Sebastian makes
implement composite psuedolikelihood

check if dropout is along what axis in structured transformer (i.e. does it just dropout random values or entire edges)

implementation for complexes (distance between residues in diff chains is inf)
    data augmentation: randomize chain assignments, or make chain identity invariant
    fine-tuning on specific interfaces i.e. antibodies
    idea: feed extra data e.g. net knows if constituients need to be soluable on their own

at some point
    look into CATH train/test splits, probably towards the end

backup files to c3ddb

measure perplexity (***)

increase model complexity to see how it does, or just feed it more data

update google slides architectures diagram
add regularization?

if i split a protein into two, does it still output a very similar potts model?

prioritize running the model on more data!!!

include target phi,psi,omega,env for native protein
add gaussian noise to coordinates as data augmentation?
