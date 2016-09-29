import sasmol.sasmol as sasmol
import numpy as np
import os
import os.path as op
import errno

def mkdir_p(path):
    '''
    make directory recursively
    adapted from http://stackoverflow.com/questions/600268/
    '''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and op.isdir(path):
            pass
        else:
            raise

pdb_fname = 'centered_mab.pdb'
run_dir = 'output'
n_dcds_list = [50, 500, 1000, 2000, 3000, 4500]

mol = sasmol.SasMol(0)
mol.read_pdb(pdb_fname)
dcd_fnames = []
cluster_out_files = []

for n_dcds in n_dcds_list:
    this_run = '{}_{}'.format(run_dir, n_dcds)
    mkdir_p(this_run)

    print('creating {} dcds'.format(n_dcds))
    for i in xrange(n_dcds):
        index = i+1
        n_print = np.int(n_dcds/10)

        if not index%n_print:
            print(index)


        this_fname = op.join(this_run, '{}_c{:05d}.dcd'.format('prefix', i+1))
        dcd_fnames.append(this_fname)
        cluster_out_files.append(mol.open_dcd_write(dcd_fnames[i]))

    for cluster_out_file in cluster_out_files:
        mol.write_dcd_step(cluster_out_file, 0, 1)
        mol.close_dcd(cluster_out_file)
