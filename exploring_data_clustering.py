
# coding: utf-8
'''
# # Exploring Data Clustering
#
# Approaches to explore:
Priciple Component Analysis
https://en.wikipedia.org/wiki/Principal_component_analysis
http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_3d.html

# Hierarchical Clustering
https://en.wikipedia.org/wiki/Hierarchical_clustering
http://scikit-learn.org/stable/modules/clustering.html

DBSCAN
http://scikit-learn.org/stable/modules/clustering.html#dbscan

There are two parameters to the algorithm, `min_samples` and `eps`,
which define formally what we mean when we say dense. Higher `min_samples`
or lower `eps` indicate higher density necessary to form a cluster.

More formally, we define a core sample as being a sample in the dataset
such that there exist `min_samples` other samples within a distance of
`eps`, which are defined as neighbors of the core sample. This tells us
that the core sample is in a dense area of the vector space. A cluster
is a set of core samples, that can be built by recursively by taking a
core sample, finding all of its neighbors that are core samples, finding
all of their neighbors that are core samples, and so on. A cluster also
has a set of non-core samples, which are samples that are neighbors of
a core sample in the cluster but are not themselves core samples.
Intuitively, these samples are on the fringes of a cluster.

Any core sample is part of a cluster, by definition. Further, any cluster
has at least min_samples points in it, following the definition of a core
sample. For any sample that is not a core sample, and does have a distance
higher than eps to any core sample, it is considered an outlier by the
algorithm.
'''

# Import and manipulate SAS data

import glob
import errno
import os
import os.path as op
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
# from sklearn import metrics
import sasmol.sasmol as sasmol


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


# In[2]:


# ~~~~~ Tune these parameters to adjust clustering ~~~~~ #

distance = 0.5   # radius in hyerdimensional space
min_samples = 2  # minimum number of structures to form a cluster
q_max = 0.2    # maximum q value for clustering

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

out_dir = 'output'
mkdir_p(out_dir)
sas_dir = 'sascalc_all'
saxs_dir = 'xray'
sans_dir = 'neutron_D2Op_100'
sas_ext = '*.iq'
saxs_search = op.join(sas_dir, saxs_dir, sas_ext)
sans_search = op.join(sas_dir, sans_dir, sas_ext)
print('looking for SAXS files in: {}'.format(saxs_search))
print('looking for SANS files in: {}'.format(sans_search))


# In[3]:
# find the SAS files
saxs_files = glob.glob(saxs_search)
sans_files = glob.glob(sans_search)
saxs_files.sort()
sans_files.sort()
print('found {} SAXS files'.format(len(saxs_files)))
print('found {} SAXS files'.format(len(sans_files)))


# In[4]:
# load in the SAXS data
print('loading in saxs data')
saxs_data = []
first_data = np.loadtxt(saxs_files[0])
q_mask = first_data[:, 0] <= q_max
first_data = first_data[q_mask]
saxs_data.append(first_data[1:, 1])
for saxs_file in saxs_files[1:]:
    x_data = np.loadtxt(saxs_file)
    x_data = x_data[q_mask]
    assert np.allclose(x_data[0, 1], first_data[0, 1]), (
        'ERROR: data not normalized')
    assert np.allclose(x_data[:, 0], first_data[:, 0]), (
        'ERROR: data not on same Q-grid')
    saxs_data.append(x_data[1:, 1])
saxs_data = np.array(saxs_data)


# In[5]:
# load in the SANS data
print('loading in sans data')
sans_data = []
first_data = np.loadtxt(sans_files[0])
q_mask = first_data[:, 0] <= q_max
first_data = first_data[q_mask]
sans_data.append(first_data[1:, 1])
for sans_file in sans_files[1:]:
    n_data = np.loadtxt(sans_file)
    n_data = n_data[q_mask]
    assert np.allclose(n_data[0, 1], first_data[0, 1]), (
        'ERROR: data not normalize')
    assert np.allclose(n_data[:, 0], first_data[:, 0]), (
        'ERROR: data not on same Q-grid')
    sans_data.append(n_data[1:, 1])
sans_data = np.array(sans_data)


# In[6]:

# q_saxs = x_data[1:, 0]
# q_sans = n_data[1:, 0]
# print(q_saxs)
# print(q_sans)


# In[7]:

# n_samples, n_features = saxs_data.shape # should be (n_samples, n_features)
# print('samples: {}\nfeatures: {}'.format(n_samples, n_features))


# In[8]:

# print(saxs_data[:5, :5])
# print(sans_data[:5, :5])


# In[9]:

# min_vals = saxs_data.min(axis=0)
# max_vals = saxs_data.max(axis=0)
# saxs_range = max_vals - min_vals
# print(saxs_range)


# In[10]:

# min_vals = sans_data.min(axis=0)
# max_vals = sans_data.max(axis=0)
# sans_range = max_vals - min_vals
# print(sans_range)


# #### Rescale the data
# Originally used `StandardScaler` but changed to `RobustScaler` to avoid
# complications from outliers (which skew the mean)

# In[11]:

print('scaling data')
x_scaler = RobustScaler()
n_scaler = RobustScaler()


# In[12]:

x_scaler.fit(saxs_data)
n_scaler.fit(sans_data)


# In[14]:

scaled_saxs = x_scaler.transform(saxs_data)
scaled_sans = n_scaler.transform(sans_data)
# print(scaled_saxs[:5, :5])
# print(scaled_sans[:5, :5])


# In[15]:

# min_vals = scaled_saxs.min(axis=0)
# max_vals = scaled_saxs.max(axis=0)
# x_scaled_range = max_vals - min_vals

# min_vals = scaled_sans.min(axis=0)
# max_vals = scaled_sans.max(axis=0)
# n_scaled_range = max_vals - min_vals

# print(x_scaled_range)
# print(n_scaled_range)


'''
# commenting out the plots

# In[16]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[17]:

plt.figure()
plt.plot(saxs_data[:,0], saxs_data[:,5], 'bo', markersize=5)
plt.title('original data')
plt.figure()
plt.plot(scaled_saxs[:,0], scaled_saxs[:,5], 'bo', markersize=5)
plt.title('scaled data')


# In[18]:

plt.figure()
plt.plot(saxs_data[:,0], saxs_data[:,15], 'bo', markersize=5)
plt.title('original data')
plt.figure()
plt.plot(scaled_saxs[:,0], scaled_saxs[:,17], 'bo', markersize=5)
plt.xlabel(r'$Q_{}$'.format(0))
plt.ylabel(r'Q_{}'.format(17))
plt.title('scaled data')


# In[19]:

i0 = 2
i_compare = 0
for i0 in range(40):
    plt.figure()
    plt.plot(scaled_saxs[:,i0], scaled_saxs[:, i_compare], 'bo')
    plt.plot(scaled_saxs[112,i0], scaled_saxs[112, i_compare], 'rs')
    plt.plot(scaled_saxs[113,i0], scaled_saxs[113, i_compare], 'gs')
    plt.xlabel(r'$Q_{}$'.format(i0))
    plt.ylabel(r'$Q_{}$'.format(i_compare))
'''

# ### DBSCAN

# In[20]:


# In[21]:

# Compute DBSCAN
# ## Tune these parameters to adjust cluster size ##
# distance = 1
# min_samples = 2
# ##################################################
print('performing DBSCAN clustering')
x_db = DBSCAN(eps=distance, min_samples=min_samples).fit(scaled_saxs)
x_core_samples_mask = np.zeros_like(x_db.labels_, dtype=bool)
x_core_samples_mask[x_db.core_sample_indices_] = True
x_labels = x_db.labels_ + 1  # 0's are independent groups
x_clusters_ = len(set(x_labels)) - (1 if -1 in x_labels else 0)

n_db = DBSCAN(eps=distance, min_samples=min_samples).fit(scaled_saxs)
n_core_samples_mask = np.zeros_like(n_db.labels_, dtype=bool)
n_core_samples_mask[n_db.core_sample_indices_] = True
n_labels = n_db.labels_ + 1  # 0's are independent groups
n_clusters_ = len(set(n_labels)) - (1 if -1 in n_labels else 0)


# In[37]:

# x-ray clusters
x_unique = set(x_labels)
x_unique.remove(0)
print('cluster labels: {}'.format(x_unique))
print('unique clusters: {}'.format(len(x_unique) + list(x_labels).count(0)))
if len(x_labels) < 200:
    for c in set(x_labels):
        print('{}: {}'.format(c, list(x_labels).count(c)))


# In[61]:

# neutron clusters
unique = set(n_labels)
unique.remove(0)
total_clusters = len(unique) + list(n_labels).count(0)
print('cluster labels: {}'.format(unique))
print('unique clusters: {}'.format(total_clusters))
if len(x_labels) < 200:
    for c in set(n_labels):
        print('{}: {}'.format(c, list(n_labels).count(c)))


# In[40]:

np.savetxt(op.join(out_dir, 'x_clusters.txt'), x_labels, fmt='%d')
np.savetxt(op.join(out_dir, 'n_clusters.txt'), n_labels, fmt='%d')
print('cluster info saved to {}/x_clusters.txt and {}/n_clusters.txt'.format(
    out_dir, out_dir))


# In[41]:

# print(n_labels)
# print(n_labels.shape)
# slabels = np.array(n_labels, dtype='str')
# print(slabels)
# print(slabels.shape)


# In[43]:
'''
# commenting out plots
i_compare = 0

mn = scaled_saxs.min(axis=0)
mx = scaled_saxs.max(axis=0)

# for i0 in range(1):
for i0 in range(40):
    plt.figure()

    # plot points to make the correct box size
    plt.plot(mn[i0], mn[i_compare], 'w.')
    plt.plot(mx[i0], mx[i_compare], 'w.')

    for j in range(len(scaled_saxs)):
        if slabels[j] != '0':
            plt.text(scaled_saxs[j, i0], scaled_saxs[j, i_compare], slabels[j],
                     fontdict={'weight': 'bold', 'size': 15},
                     color='r') # plt.cm.Set1(labels[i]/10.0))
        else:
            plt.plot(scaled_saxs[j, i0], scaled_saxs[j, i_compare], 'k.',
                    markersize=5)

    plt.xlabel(r'$Q_{}$'.format(i0))
    plt.ylabel(r'$Q_{}$'.format(i_compare))
'''

# ### Write DCD output

# In[87]:

print('opening dcd file for structure extraction')
dcd_fname = glob.glob('*.dcd')
assert len(dcd_fname) == 1, 'ERROR: multiple dcd files: {}'.format(dcd_fname)
dcd_fname = dcd_fname[0]

print('opening pdb file for atom information')
pdb_fname = glob.glob('*.pdb')
assert len(pdb_fname) == 1, 'ERROR: multiple pdb files: {}'.format(pdb_fname)
pdb_fname = pdb_fname[0]


# In[88]:

mol = sasmol.SasMol(0)
mol.read_pdb(pdb_fname)


# In[89]:

if not np.alltrue(n_labels == x_labels):
    print('WARNING: labels do not match\nusing neutron labels')
labels = n_labels


# In[92]:

# create a dcd for every cluster with >1 frame
print('opening dcd for for storing unique structures')
dcd_fnames = []
cluster_out_files = []  # dcds for clusters
unique_out_fname = op.join(out_dir, '{}_uniue.dcd'.format(dcd_fname[:-4]))
dcd_out_file = mol.open_dcd_write(unique_out_fname)

dcd_in_file = mol.open_dcd_read(dcd_fname)

print('opening {} dcds for the clustered structres'.format(len(unique)))
for i in xrange(len(unique)):
    this_fname = op.join(out_dir, '{}_c{:05d}.dcd'.format(dcd_fname[:-4], i+1))
    dcd_fnames.append(this_fname)
    cluster_out_files.append(mol.open_dcd_write(dcd_fnames[i]))

visited_cluster = set()
dcd_out_frame = 0
cluster_out_frame = np.zeros(len(unique), dtype=int)

print('populating dcd files with structures')
for (i, label) in enumerate(labels):
    mol.read_dcd_step(dcd_in_file, i)
    if label == 0:
        dcd_out_frame += 1
        mol.write_dcd_step(dcd_out_file, 0, dcd_out_frame)
    else:
        cluster_out_frame[label-1] += 1
        # print('adding frame to cluster {}'.format(label-1))
        # print(cluster_out_frame)
        mol.write_dcd_step(cluster_out_files[label-1], 0,
                           cluster_out_frame[label-1])
        if label not in visited_cluster:
            visited_cluster.add(label)
            dcd_out_frame += 1
            mol.write_dcd_step(dcd_out_file, 0, dcd_out_frame)

print('closing dcd files')
for cluster_out_file in cluster_out_files:
    mol.close_dcd_write(cluster_out_file)

mol.close_dcd_write(dcd_out_file)
mol.close_dcd_read(dcd_in_file[0])
