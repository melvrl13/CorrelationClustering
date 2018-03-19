#!/usr/bin/env python

# Currently works with protein CA onlyo

import mdtraj
import numpy
import hdbscan
import pandas
import signal
import subprocess
import os

# Class declarations
class Selector:
    def __init__(self, trajectory, atom_selection):
        assert isinstance(trajectory, mdtraj.Trajectory)
        self.trajectory = trajectory
        self.sel = atom_selection

    def select(self):
        raise NotImplementedError


class Slice(Selector):
    def select(self):
        indices = self.trajectory.top.select(self.sel)
        sub_trajectory = self.trajectory.atom_slice(atom_indices=indices, inplace=False)
        return sub_trajectory


class Reader:
    def __init__(self, trajectory_path):
        self.trajectory = trajectory_path

    def load(self):
        raise NotImplementedError


class DCD(Reader):
    def __init__(self, trajectory_path, topology_path):
        self.topology = topology_path
        super().__init__(trajectory_path)

    def load(self):
        trajectory = mdtraj.load(self.trajectory, top=self.topology)
        return trajectory


class Correlation:
    def __init__(self, trajectory):
        assert isinstance(trajectory, mdtraj.Trajectory)
        self.trajectory = trajectory
        self.correlation_matrix = []

    def calculate(self):
        raise NotImplementedError


class Pearson(Correlation):
    def calculate(self):
        average = numpy.average(self.trajectory.xyz, axis=0)
        fluctuations = self.trajectory.xyz - average[numpy.newaxis, :]
        del average
        dots = numpy.zeros((self.trajectory.n_atoms, self.trajectory.n_atoms))
        for i in range(self.trajectory.n_frames):
            dot = numpy.dot(fluctuations[i, :, :], numpy.transpose(fluctuations[i, :, :]))
            dots = dots + dot
        del fluctuations
        dots = numpy.divide(dots, self.trajectory.n_frames)
        diagonal = numpy.diag(dots)
        normalization_matrix = numpy.outer(diagonal, diagonal)
        normalization_matrix = numpy.sqrt(normalization_matrix)
        self.correlation_matrix = numpy.divide(dots, normalization_matrix)
        return self.correlation_matrix


class TimeLagged(Correlation):
    def __init__(self, trajectory, covariance_tau):
        assert isinstance(covariance_tau, int)
        self.covariance_tau=covariance_tau
        self.normalization_matrix = []
        super().__init__(trajectory)
    def calculate(self):
        average = numpy.average(self.trajectory.xyz, axis=0)
        fluctuations = self.trajectory.xyz - average[numpy.newaxis, :]
        del average
        dots = numpy.zeros((self.trajectory.n_atoms, self.trajectory.n_atoms))
        for i in range(self.trajectory.n_frames - self.covariance_tau):
            dot = numpy.dot(fluctuations[i, :, :], numpy.transpose(fluctuations[i + self.covariance_tau, :, :]))
            dots = dots + dot
        del fluctuations
        dots = numpy.divide(dots, self.trajectory.n_frames)
        diagonal = numpy.diag(dots)
        self.normalization_matrix = numpy.outer(diagonal, diagonal)
        self.normalization_matrix = numpy.sqrt(numpy.absolute(self.normalization_matrix))
        self.correlation_matrix = numpy.divide(dots, self.normalization_matrix)
        return self.correlation_matrix

class Saver:
    def __init__(self, out_name):
        self.out_name = out_name

    def save(self):
        raise NotImplementedError


class ClusterFrames(Saver):
    def __init__(self, out_name, labels):
        self.labels = labels
        super().__init__(out_name)
        num_frames = len(self.labels)
        self.clusters = int(max(self.labels)) + 1
        self.labeled_traj = pandas.DataFrame(columns=['frame', 'cluster'])
        self.labeled_traj['frame'] = numpy.arange(num_frames)
        self.labeled_traj['cluster'] = self.labels

    def save(self):
        with open(self.out_name, 'w') as f:
            for i in range(0, self.clusters):
                cluster_string = ' '.join(
                        ['%d' % num for num in self.labeled_traj.loc[self.labeled_traj['cluster'] == i].frame.values]
                        )
                f.write(cluster_string + '\n')


class Clustering:
    @staticmethod
    def cluster(correlation_matrix, input_type='correlation', minimum_membership=None):
        number_residues = len(correlation_matrix)
        three_percent = int(numpy.ceil(number_residues * 0.03))
        if minimum_membership:
            min_cluster_size = minimum_membership
        elif three_percent >= 2:
            min_cluster_size = three_percent
        else:
            min_cluster_size = 2

        if input_type == 'similarity':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
            distance = 1 - numpy.abs(correlation_matrix)
            labels = clusterer.fit_predict(distance)
        elif input_type == 'dots':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
            dots = correlation_matrix.dot(correlation_matrix.T)
            distance = 1 - dots / numpy.max(numpy.abs(dots))
            labels = clusterer.fit_predict(distance)
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(correlation_matrix)

        return labels

    @staticmethod
    def visualize(labels, pdb_file, out_name):
        num_residues = len(labels)
        df = pandas.DataFrame(columns=['residue', 'cluster'])
        df['residue'] = numpy.arange(num_residues)
        df['cluster'] = labels
        clusters = numpy.unique(df.loc[df['cluster'] != -1].cluster.values)
        noise_string = ' '.join(
                ['%d' % num for num in df.loc[df['cluster'] == -1].residue.values]
                )

        with open(out_name + '_image_script.vmd', 'w+') as vmd_file:
            vmd_file.write(
                    'mol new ' + pdb_file + '\n' + 'mol delrep 0 top\n')
            if len(df.loc[df['cluster'] == -1]) > 0:
                vmd_file.write(
                        'mol representation NewCartoon\n'
                        + 'mol selection {residue ' + noise_string + '}\n' + 'mol material Ghost\n' + 'mol addrep top\n')
                vmd_file.write(
                        'mol representation NewCartoon\n' + 'mol material AOChalky\n' + 'display ambientocclusion on\n')
            for cluster in clusters:
                cluster_string = ' '.join(
                        ['%d' % num for num in df.loc[df['cluster'] == cluster].residue.values]
                        )
                vmd_file.write(
                        'mol color ColorID ' + str(cluster) + '\n' + 'mol selection {residue ' + cluster_string + '}\n'
                        + 'mol addrep top\n'
                        )
            vmd_file.write(
                    'display resize 1920 1080\n' + 'display resetview\n' + 'render TachyonInternal ' + out_name + '\nexit')

                    # Posix-compliant way to exit shell
        signal.signal(signal.SIGTTOU, signal.SIG_IGN)
        # Now, let's make some pretty pictures
        vmd_render_cmd = (
                'vmd '
                + ' -dispdev text -e '
                + out_name + '_image_script.vmd'
                )
        subprocess.call([os.getenv('SHELL'), '-i', '-c', vmd_render_cmd])
        os.tcsetpgrp(0, os.getpgrp())



if __name__ == "__main__":

    import argparse

    # Initialize parser. The default help has poor labeling. See http://bugs.python.org/issue9694
    parser = argparse.ArgumentParser(description='Perform correlation clustering and generate images', add_help=False)

    # List all possible user input
    inputs = parser.add_argument_group('Input arguments')
    inputs.add_argument('-h', '--help', action='help')
    inputs.add_argument('-pdb',
            action='store',
            dest='structure',
            help='PDB corresponding to trajectory. MUST be PDB for this script.',
            type=str,
            required=True
            )
    inputs.add_argument('-traj',
            action='store',
            dest='trajectory',
            help='Trajectory',
            type=str,
            required=True
            )
    inputs.add_argument('-sel',
            action='store',
            dest='sel',
            help='Atom selection. Must be name CA currently.',
            type=str,
            default='name CA'
            )
    inputs.add_argument('-tau',
            action='store',
            dest='covariance_tau',
            default=None,
            type=int,
            help='Lag time for constructing a time-lagged correlation matrix',
            )
    inputs.add_argument('-m',
            action='store',
            dest='min_cluster_size',
            help='Minimum cluster and neighborhood size for HDBSCAN',
            type=int
            )
    # maybe make the dot product flag -dot
    inputs.add_argument('-d',
            action='store_true',
            dest='correlation_as_distance',
            help='Use 1 minus abs(correlation) matrix as distance',
            )
    inputs.add_argument('-p',
            action='store_true',
            dest='dot_products_as_distance',
            help='Use dot products as similarity matrix.',
            )
    inputs.add_argument('-o',
            action='store',
            dest='out_name',
            help='Output prefix for vmd files and image',
            type=str,
            required=True
            )

    # Parse into useful form
    UserInput = parser.parse_args()

    # Process trajectory
    trajectory = DCD(topology_path=UserInput.structure, trajectory_path=UserInput.trajectory).load()
    trajectory = Slice(trajectory=trajectory, atom_selection=UserInput.sel).select()

    if UserInput.correlation_as_distance:
        input_type = 'similarity'
    elif UserInput.dot_products_as_distance:
        input_type = 'dots'
    else:
        input_type = 'correlation'

    if UserInput.min_cluster_size:
        min_membership = UserInput.min_cluster_size
    else:
        min_membership = None

    if UserInput.covariance_tau:
        correlation_matrix = TimeLagged(
                trajectory=trajectory, covariance_tau=UserInput.covariance_tau
                ).calculate()
    else:
        correlation_matrix = Pearson(trajectory=trajectory).calculate()
    labels = Clustering.cluster(
            correlation_matrix=correlation_matrix, input_type=input_type, minimum_membership=min_membership
            )
    ClusterFrames(out_name=UserInput.out_name + '_residue_groups.txt', labels=labels).save()
    Clustering.visualize(
            labels=labels, pdb_file=UserInput.structure, out_name=UserInput.out_name + '_render.tga'
            )
