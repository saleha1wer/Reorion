# Olion
A re-implementation of Orion (https://cbw.sh/static/pdf/orion-wosn10.pdf) with some additional landmark selection strategies.

## Requirments
* Python 3.8.2+

## Packages
* Numpy
* NetworkX
* SciPy

## Network sources 
* Authors: http://snap.stanford.edu/data/com-DBLP.html
* GitHub: https://snap.stanford.edu/data/github-social.html
* Gowalla: http://konect.cc/networks/loc-gowalla_edges/

## Running instructions
In terminal or CMD, run: 
<br />
<br />
`python Olion.py <path-to-edge-list>`
<br />
<br />
The default landmark selection strategy is random, to run with a specific strategy:
<br />
<br />
`python Olion.py <path-to-edge-list> <strategy>`
<br />
<br />
Strategy is one of --> [random, degree, max_ave_distance, max_min_distance, convex_hull]
<br />
<br />
This is for running on a TXT edge list. For one in CSV format, see Testing Scripts/olion_GitHub.py <br />
<br />
This returns a .npy file, containing a dictionary of (node, coordinates) pairs, in the same directory as the edge list. 
<br /> <br />

