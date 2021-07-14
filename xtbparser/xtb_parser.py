#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import numpy as np
import logging
from ase.io import read as aseread
from ase import Atoms as aseAtoms

from nomad.units import ureg
from nomad.parsing.parser import FairdiParser
from nomad.parsing.file_parser import Quantity, TextParser
from nomad.datamodel.metainfo.run import Run
from nomad.datamodel.metainfo.method import TB, TBModel, Interaction
from nomad.datamodel.metainfo.system import System, Atoms
from nomad.datamodel.metainfo.calculation import SingleConfigurationCalculation,\
    ScfIteration, Energy


class OutParser(TextParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_quantities(self):
        re_f = r'[\d\.E\+\-]+'

        def str_to_eigenvalues(val_in):
            occupations, energies = [], []
            for val in val_in.strip().split('\n'):
                val = val.split('(')[0].split()
                if not val[0].isdecimal():
                    continue
                occupations.append(float(val.pop(1)) if len(val) > 3 else 0.0)
                energies.append(float(val[1]))
            return occupations, energies * ureg.hartree

        def str_to_parameters(val_in):
            val = [v.strip() for v in val_in.split('  ', 1)]
            val[1] = val[1].split()
            return val

        common_quantities = [
            Quantity(
                'setup',
                r'SETUP\s*:\s*([\s\S]+?\.+\n *\n)',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'parameters',
                        r'\n +\: +(.+?\s{2,}[\w\.\-\+]+)', str_operation=lambda x: [
                            v.strip() for v in x.split('  ', 1)], repeats=True)])),
            Quantity(
                'summary',
                r'(SUMMARY[\s\S]+?\:\n *\n)',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'energy_total',
                        rf':: total energy\s*({re_f})',
                        unit=ureg.hartree, dtype=np.float64),
                    Quantity(
                        'gradient_norm',
                        rf':: gradient norm\s*({re_f})',
                        unit=ureg.hartree / ureg.angstrom, dtype=np.float64),
                    Quantity(
                        'hl_gap',
                        rf':: HOMO-LUMO gap\s*({re_f})',
                        unit=ureg.eV, dtype=np.float64),
                    Quantity(
                        'energy_scc',
                        rf':: SCC energy\s*({re_f})',
                        unit=ureg.hartree, dtype=np.float64),
                    Quantity(
                        'energy_bond',
                        rf':: bond energy\s*({re_f})',
                        unit=ureg.hartree, dtype=np.float64),
                    Quantity(
                        'energy_repulsion',
                        rf':: repulsion energy\s*({re_f})',
                        unit=ureg.hartree, dtype=np.float64),
                    Quantity(
                        'energy_electrostatic',
                        rf':: electrostat energy\s*({re_f})',
                        unit=ureg.hartree, dtype=np.float64),
                    Quantity(
                        'charge_total',
                        rf':: total charge\s*({re_f})',
                        unit=ureg.elementary_charge, dtype=np.float64)]))]

        orbital_quantities = [
            Quantity(
                'eigenvalues',
                r'# +Occupation +Energy.+\s*\-+([\s\S]+?)\-+\n',
                str_operation=str_to_eigenvalues),
            Quantity(
                'hl_gap',
                rf'HL\-Gap\s*({re_f})', dtype=np.float64, unit=ureg.hartree),
            Quantity(
                'energy_fermi',
                rf'Fermi\-level\s*({re_f})', dtype=np.float64, unit=ureg.hartree)]

        property_quantities = orbital_quantities + [
            Quantity(
                'dipole',
                r'(dipole:[\s\S]+?)molecular',
                sub_parser=TextParser(quantities=[
                    Quantity('q', r'q only:(.+)', dtype=np.dtype(np.float64)),
                    Quantity('full', r'full:(.+)', dtype=np.dtype(np.float64))])),
            Quantity(
                'quadrupole',
                r'(quadrupole \(traceless\):[\s\S]+?)\n *\n',
                sub_parser=TextParser(quantities=[
                    Quantity('q', r'q only:(.+)', dtype=np.dtype(np.float64)),
                    Quantity('full', r'full:(.+)', dtype=np.dtype(np.float64)),
                    Quantity('q_dip', r'q\+dip:(.+)', dtype=np.dtype(np.float64))]))]

        geometry_quantities = [
            Quantity('file', r'optimized geometry written to:\s*(\S+)')]

        scf_quantities = common_quantities + orbital_quantities + [
            Quantity(
                'model',
                r'(Reference\s*[\s\S]+?\n *\n)',
                sub_parser=TextParser(quantities=[
                    Quantity('reference', r'Reference\s*(\S+)'),
                    Quantity(
                        'contribution',
                        r'(\w+:\s*[\s\S]+?)(?:\*|\n *\n)',
                        repeats=True, sub_parser=TextParser(quantities=[
                            Quantity('name', r'(\w+):'),
                            Quantity(
                                'parameters',
                                r'\n +(\w.+?  .+)',
                                str_operation=str_to_parameters, repeats=True)
                        ]))])),
            Quantity(
                'scf_iteration',
                r'iter\s*E\s*dE.+([\s\S]+?convergence.+)',
                sub_parser=TextParser(quantities=[
                    Quantity('step', r'(\d+ .+)', repeats=True),
                    Quantity(
                        'converged',
                        r'(\*\*\* convergence criteria.+)',
                        str_operation=lambda x: 'satisfied' in x)])),
        ]

        optimization_quantities = [
            Quantity(
                'cycle',
                r'CYCLE +\d([\s\S]+?\n *\n)',
                repeats=True, sub_parser=TextParser(quantities=[
                    Quantity(
                        'energy_total',
                        rf'total energy +: +({re_f})',
                        dtype=np.float64, unit=ureg.hartree),
                    Quantity(
                        'scf_iteration',
                        rf'\.+\s+(\d+\s+{re_f}[\s\S]+?)\*',
                        sub_parser=TextParser(quantities=[
                            Quantity('step', r'(\d+ .+)', repeats=True),
                            Quantity('time', rf'SCC iter\. +\.+ +(\d+) min, +({re_f}) sec')
                        ]))])),
            Quantity(
                'converged',
                r'(\*\*\* GEOMETRY OPTIMIZATION.+)',
                str_operation=lambda x: 'CONVERGED' in x),
            Quantity(
                'final_structure',
                r'final structure:([\s\S]+?\-+\s+\|)',
                sub_parser=TextParser(quantities=[
                    Quantity('atom_labels', r'([A-Z][a-z]?) ', repeats=True),
                    Quantity(
                        'atom_positions',
                        rf'({re_f} +{re_f} +{re_f})',
                        unit=ureg.angstrom, dtype=np.dtype(np.float64))])),
            Quantity(
                'final_scf',
                r'(Final Singlepoint +\|[\s\S]+?::::::::::::)',
                sub_parser=TextParser(quantities=scf_quantities))

        ] + common_quantities

        md_quantities = [
            Quantity(
                'traj_file',
                r'trajectories on (.+?\.trj)'
            )
        ]

        self._quantities = [
            Quantity('program_version', r'\* xtb version ([\d\.]+)'),
            Quantity(
                'calculation_setup',
                r'Calculation Setup +\|\s*\-+\s*([\s\S]+?)\-+\s+\|',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'parameters', r'([\w ]+:.+)',
                        str_operation=lambda x: [v.strip() for v in x.split(':')], repeats=True)])),
            Quantity(
                'gfnff',
                r'(G F N - F F[\s\S]+?::::::::::::\n *\n)',
                sub_parser=TextParser(quantities=scf_quantities)),
            Quantity(
                'gfn1',
                r'(G F N 1 - x T B[\s\S]+?::::::::::::\n *\n)',
                sub_parser=TextParser(quantities=scf_quantities)),
            Quantity(
                'gfn2',
                r'(G F N 2 - x T B[\s\S]+?::::::::::::\n *\n)',
                sub_parser=TextParser(quantities=scf_quantities)),
            Quantity(
                'ancopt',
                r'(A N C O P T +\|[\s\S]+?::::::::::::\n *\n)',
                sub_parser=TextParser(quantities=optimization_quantities)),
            Quantity(
                'md',
                r'(Molecular Dynamics +\|[\s\S]+?exit of md)',
                sub_parser=TextParser(quantities=md_quantities)),
            Quantity(
                'property',
                r'(Property Printout +\|[\s\S]+?\-+\s+\|)',
                sub_parser=TextParser(quantities=property_quantities)),
            Quantity(
                'geometry',
                r'(Geometry Summary +\|[\s\S]+?\-+\s+\|)',
                sub_parser=TextParser(quantities=geometry_quantities)),
            Quantity(
                'energy_total', rf'\| TOTAL ENERGY\s*({re_f})',
                dtype=np.float64, unit=ureg.hartree),
            Quantity(
                'gradient_norm',
                rf'\| GRADIENT NORM\s*({re_f})',
                dtype=np.float64, unit=ureg.hartree / ureg.angstrom),
            Quantity(
                'hl_gap',
                rf'\| HOMO-LUMO GAP\s*({re_f})',
                dtype=np.float64, unit=ureg.eV),
            Quantity(
                'topo_file',
                r'Writing topology from bond orders to (.+\.mol)'),
            Quantity(
                'footer',
                r'(\* finished run on [\s\S]+?\Z)',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'end_time',
                        r'finished run on (\S+) at (\S+)', flatten=False),
                    Quantity(
                        'wall_time',
                        r'\* +wall-time: +(\d+) d, +(\d+) h, +(\d+) min, +([\d\.]+) sec',
                        repeats=True),
                    Quantity(
                        'cpu_time',
                        r'\* +cpu-time: +(\d+) d, +(\d+) h, +(\d+) min, +([\d\.]+) sec',
                        repeats=True)]))]


class CoordParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        re_f = r'[\d\.\-]+'

        self._quantities = [
            Quantity('coord_unit', r'\$coord(.+)'),
            Quantity(
                'positions_labels',
                rf'({re_f} +{re_f} +{re_f} +[A-Za-z]+\s+)', repeats=True),
            Quantity('periodic', r'\$periodic(.+)'),
            Quantity('lattice_unit', r'\$lattice(.+)'),
            Quantity(
                'lattice',
                rf'({re_f} +{re_f} +{re_f}) *\n', repeats=True, dtype=np.dtype(np.float64)),
            Quantity('cell_unit', r'\$cell(.+)'),
            Quantity(
                'cell',
                rf'({re_f} +{re_f} +{re_f} +{re_f} +{re_f} +{re_f}) *\n',
                dtype=np.dtype(np.float64))
        ]

    def get_atoms(self):
        positions = self.get('positions_labels')
        if positions is None:
            return

        lattice_unit = self.get('lattice_unit', '').strip()
        lattice_unit = ureg.angstrom if lattice_unit.startswith('angs') else ureg.bohr
        lattice = self.get('lattice')
        lattice = (lattice * lattice_unit).to('angstrom').magnitude if lattice is not None else lattice

        cell = self.get('cell')
        if cell is not None:
            cell_unit = self.get('cell_unit')
            cell_unit = ureg.angstrom if cell_unit is not None else ureg.bohr
            cell_abc = (cell[:3] * cell_unit).to('angstrom').magnitude
            lattice = list(cell_abc) + list(cell[3:])

        labels = [p[-1].title() for p in positions]
        positions = [p[:3] for p in positions]
        coord_unit = self.get('coord_unit', '').strip()
        if coord_unit.startswith('frac') and lattice is not None:
            positions = np.dot(positions, lattice)
        elif coord_unit.startswith('angs'):
            positions = positions * ureg.angstrom
        else:
            positions = positions * ureg.bohr
        positions = positions.to('angstrom').magnitude

        pbc = ([True] * int(self.get('periodic', 0))) + [False] * 3

        return aseAtoms(symbols=labels, positions=positions, cell=lattice, pbc=pbc[:3])


class TrajParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        re_f = r'[\d\.\-]+'

        self._quantities = [
            Quantity(
                'frame',
                r'energy\:([\s\S]+?(?:\Z|\n *\d+ *\n))',
                repeats=True, sub_parser=TextParser(quantities=[
                    Quantity(
                        'positions',
                        rf'({re_f} +{re_f} +{re_f})',
                        repeats=True, dtype=np.dtype(np.float64)),
                    Quantity('labels', r'\n *([A-Za-z]{1,2}) +', repeats=True)]))]

    def get_atoms(self, n_frame):
        frame = self.get('frame')[n_frame]
        return aseAtoms(symbols=frame.labels, positions=frame.positions)


class XTBParser(FairdiParser):
    def __init__(self):
        super().__init__(
            name='parsers/xtb', code_name='XTB',
            code_homepage='https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb',
            mainfile_contents_re=(
                r'\s*-----------------------------------------------------------\s*'
                r'\s*\|                   =====================                   \|\s*'
                r'\s*\|                           x T B                           \|\s*'
                r'\s*\|                   =====================                   \|\s*'
            ))
        self.out_parser = OutParser()
        self.coord_parser = CoordParser()
        self.traj_parser = TrajParser()

    def init_parser(self):
        self.out_parser.mainfile = self.filepath
        self.out_parser.logger = self.logger
        self.calculation_type = None

    def parse_system(self, source):
        if isinstance(source, int):
            atoms = self.traj_parser.get_atoms(source)
        elif source.endswith('.xyz') or source.endswith('.poscar'):
            atoms = aseread(os.path.join(self.maindir, source))
        else:
            self.coord_parser.mainfile = os.path.join(self.maindir, source)
            atoms = self.coord_parser.get_atoms()

        sec_system = self.archive.section_run[0].m_create(System)
        sec_atoms = sec_system.m_create(Atoms)
        sec_atoms.labels = atoms.get_chemical_symbols()
        sec_atoms.positions = atoms.get_positions() * ureg.angstrom
        lattice_vectors = np.array(atoms.get_cell())
        if np.count_nonzero(lattice_vectors) > 0:
            sec_atoms.lattice_vectors = lattice_vectors * ureg.angstrom
            sec_atoms.periodic = atoms.get_pbc()

    def parse_calculation(self, source):
        sec_scc = self.archive.section_run[0].m_create(SingleConfigurationCalculation)
        for step in source.get('scf_iteration', {}).get('step', []):
            sec_scf = sec_scc.m_create(ScfIteration)
            sec_scf.energy_total = Energy(value=step[1] * ureg.hartree)

        energy = self.out_parser.get('energy_total')
        if energy is not None:
            sec_scc.energy_total = Energy(value=energy)

    def parse_method(self, section):
        model = self.out_parser.get(section, {}).get('model')
        if model is None:
            return

        sec_method = self.archive.section_run[0].m_create(TB)
        sec_tb_model = sec_method.m_create(TBModel)
        sec_tb_model.name = section

        if model.get('reference') is not None:
            sec_tb_model.reference = model.reference

        for contribution in model.get('contribution', []):
            if contribution.name.lower() == 'hamiltonian':
                sec_interaction = sec_tb_model.m_create(Interaction, TBModel.hamiltonian)
            elif contribution.name.lower() == 'coulomb':
                sec_interaction = sec_tb_model.m_create(Interaction, TBModel.coulomb)
            elif contribution.name.lower() == 'repulsion':
                sec_interaction = sec_tb_model.m_create(Interaction, TBModel.repulsion)
            else:
                sec_interaction = sec_tb_model.m_create(Interaction, TBModel.contributions)
            sec_interaction.parameters = {p[0]: p[1] for p in contribution.parameters}

    def parse_gfn(self, section):
        if self.out_parser.get(section) is None:
            return

        self.parse_method(section)

        coord_file = [p[1] for p in self.out_parser.get('calculation_setup', {}).get(
            'parameters', []) if p[0] == 'coordinate file']

        if coord_file:
            self.parse_system(coord_file[-1])

        self.parse_calculation(section)

        self.calculation_type = 'single_point'

    def parse_opt(self, section):
        ancopt = self.out_parser.get(section)
        if ancopt is None:
            return

        self.traj_parser.mainfile = os.path.join(self.maindir, 'xtbopt.log')

        for n, cycle in enumerate(ancopt.get('cycle', [])):
            self.parse_system(n)
            self.parse_calculation(cycle)

        self.calculation_type = 'geometry_optimization'

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.maindir = os.path.dirname(self.filepath)
        self.logger = logger if logger is not None else logging

        self.init_parser()

        sec_run = self.archive.m_create(Run)
        sec_run.program_name = 'XTB'
        sec_run.program_version = self.out_parser.get('program_version')

        if self.out_parser.gfnff is not None:
            self.parse_gfn('gfnff')
        if self.out_parser.gfn1 is not None:
            self.parse_gfn('gfn1')
        if self.out_parser.gfn2 is not None:
            self.parse_gfn('gfn2')
        if self.out_parser.ancopt is not None:
            self.parse_opt('ancopt')
