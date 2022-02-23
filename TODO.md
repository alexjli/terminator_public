Rive each residue a chain index

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

measure perplexity

increase model complexity to see how it does, or just feed it more data

update google slides architectures diagram
add regularization?

if i split a protein into two, does it still output a very similar potts model?

prioritize running the model on more data!!!

include target phi,psi,omega,env for native protein
add gaussian noise to coordinates as data augmentation?

check amino acid embeddings
see what parts of the net is necessary
check TERM matches data leakage

send vikram path to run_7000_cpl_term1_tanh run
automatic experiment recording

drop the sequence input from net 1?



energies archs
- mpnn (node then edge)
- mpnn (node, then edge is derived from 2nodes + edges through ffn)
- attention (node edge alternation)
- attention (node then edge)
- attention (node, then edge is derived from 2node + edges through ffn)
- increase energies dimensionality from 32 to 64?
- increase num layers from 3 to 6 (currently doing)
- can i get GVP up and running?

matches transformer
- embed target info in only pool token

- ablate TERM graph prop (does that actually do much?)


objective: average nlcpl over all residue pairs in a batch rather than per protein



TODO 3/24/21:
- run per-res nlcpl (in progress)
- run semi-shuffle
- run gvp
- run 6 layers h128


ablate coordinates

TODO 7/12/21:

get GVP working and run TERMinator with GVP (replace struct2seq with GVP)

ablate singleton embeddings **[finished: alexjli]**

ablate pairwise embeddings **[finished: alexjli]**

ablate TERM graph propagation (delete it and linearize it) **[finished: alexjli]**

model performance varies as a function of TERM matches

replace Potts model with most likely sequence

interpretability (integrated gradients)

=======

TODO 7/28/21:

run all ablation studies in triplicate and see inter-run variability

see performance on multichain to determine whether or not we need triplicate

make a defined split for multichain train/validate (random split and stick with it)

plot performance on test with closest percent id redundant chain in train

try gevorg`s sequence complexity penalty on some test sets to see how performance changes

prioritize gvp model!!


8/17
on ablate_s2s, look at low recovery vs high recovery structures
    apply complexity filter
    label low vs high

anneal to like 0.5

check bad outputs of gvp, see what''s wrong

9/29

see if dTERMen etabs and TERMinator etabs correlate
fine-tuning
GVP layers

10/6

journal paper ideas:
1. ablate singleton model as reference for all numbers (split across mindren/vikram)
4. GVP (depending on how much work we need to work)
4. complexity filter effect
2. compare energy tables (talking up potts model)
3. run on multichain model
-- change how chain breaks are represented

people using the model:
run without TERM match data (but with TERM struture decomposition) vs without TERM data itself
2. try running with one fake match (without retraining) and see what happens
3. try training with one fake match and all matches, test with no matches but only fake matches
1. prepare the form of the model which doesn`t use TERM data
4. try to feed in partial TERMs

compare energy tables for dTERMen, TERMinator, ablated TERMinator models
- correlation between energy tables flattened to vector (or maybe matrix correlation)
make some case of potts vs autoregressive models in journal paper
replace mpnn layers w GVP layers
test full TERMinator with 0 TERM inputs
feed in a blank 1st match with structure info but no sequence
train multichain model

get model working that runs with no TERM data

10/20
make repo config file
- docstrings
**1. compare energy tables (talking up potts model)**
- alex
2. run on multichain model
- alex (look in to)
-- change how chain breaks are represented
3. ablate singleton model as reference for all numbers (split across mindren/vikram)
- assigned to vikram/mindren
4. GVP (depending on how much work we need to work)
- alex
4. complexity filter effect
- mindren

people using the model:
run without TERM match data (but with TERM struture decomposition) vs without TERM data itself
1. prepare the form of the model which doesn`t use TERM data
- alex
2. try running with one fake match (without retraining) and see what happens
- mindren?
3. try training with one fake match and all matches, test with no matches but only fake matches
- mindren?
4. try to feed in partial TERMs
- ?

11/3 
compute average run times to add to the paper
include sebastian in acknowledgements

poster pdf for neurips
biorender / diagrams.net / latex / chimera / pymol

Alex:
1. *** scatter plot for etabs ***
- subtract mean, per site, for singleton and pairwise energies for plotting
- plotting the top 1% energies
1. residue likelihoods / energies and correlate
- can energy tables be different locally but similar performance globally
- puts everything on same scale
2. recompute feature files with 999 torsions lifted to 3 torus masked
2. run multichain (just choose a random split)
3. test suite to fix GVP + restructure
- if nothing works then hyperparameter search
4. TERMless TERMinator
- dropout TERMs


Code Cleanup notes:
delete unnecessary TERMinator classes
docstrings
tests
    yapf, pylist, pytest
    (set up so that can't commit without test)
    small unit tests for methods, integration tests for large
    test input and output shapes, deterministic computations, test it runs
Document hyperparameters in default_hyperparams.py
make sure code taken from other places is cited

throw away contact index?

take dtermen run folder directly rather than try to search for the right folder?
