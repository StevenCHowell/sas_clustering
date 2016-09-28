import sasmol.sasmol as sasmol
import numpy as np
import os.path as op

pdb_fname = 'centered_mab.pdb'
run_dir = 'output'
n_dcds = 4500
mol = sasmol.SasMol(0)
mol.read_pdb(pdb_fname)
dcd_fnames = []
cluster_out_files = []

print('creating {} dcds'.format(n_dcds))
for i in xrange(n_dcds):
    index = i+1
    n_print = np.int(n_dcds/10)

    if not index%n_print:
        print(index)


    this_fname = op.join(run_dir, '{}_c{:05d}.dcd'.format('prefix', i+1))
    dcd_fnames.append(this_fname)
    cluster_out_files.append(mol.open_dcd_write(dcd_fnames[i]))
