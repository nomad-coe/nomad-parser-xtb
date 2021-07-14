#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
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

from setuptools import setup, find_packages


def main():
    setup(
        name='xtbparser',
        version='1.0',
        description='NOMAD parser implementation for xTB.',
        author='The NOMAD Authors',
        license='APACHE 2.0',
        packages=find_packages(exclude=['tests']),
        install_requires=['nomad-lab'])


if __name__ == '__main__':
    main()
