###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to
#  submit large numbers of jobs on supercomputers. It provides a python interface to physical input,
#  such as crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential
#  programs. It is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU
#  General Public License as published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################
# -*- coding: utf-8 -*-
from pytest import fixture, mark
from pylada.espresso import Card


@fixture
def card_stream():
    from io import StringIO
    return StringIO("ATOMIC_SPECIES\nAl 1 this.that")


@fixture
def subtitled_stream():
    from io import StringIO
    return StringIO("K_POINTS tpiba\n2\n0 0 0 0.8\n0.5 0.5 0.5 0.2")


def test_create_card():
    card = Card('ATOMIC_SPECIES')
    assert card.name == 'atomic_species'
    assert card.subtitle is None
    assert card.value is None


def test_create_card_with_value_and_subtitle():
    card = Card('ATOMIC_SPECIES', 'subtitle', '2')
    assert card.name == 'atomic_species'
    assert card.subtitle == '2'
    assert card.value == 'subtitle'


def test_read_card(card_stream):
    card = Card('ATOMIC_SPECIES')
    card.read(card_stream)
    assert card.name == 'atomic_species'
    assert card.subtitle is None
    assert card.value == 'Al 1 this.that'


def test_read_subtitled_card(subtitled_stream):
    card = Card('K_POINTS')
    card.read(subtitled_stream)
    assert card.name == 'k_points'
    assert card.subtitle == 'tpiba'
    assert card.value == "2\n0 0 0 0.8\n0.5 0.5 0.5 0.2"


def test_goaround(card_stream):
    from io import StringIO
    card, read_card = Card('ATOMIC_SPECIES'), Card('ATOMIC_SPECIES')
    card.read(card_stream)
    read_card.read(StringIO(repr(card)))
    assert read_card.name == 'atomic_species'
    assert read_card.subtitle is None
    assert read_card.value == 'Al 1 this.that'


def test_goaround_subtitled(subtitled_stream):
    from io import StringIO
    card, read_card = Card('K_POINTS'), Card('K_POINTS')
    card.read(subtitled_stream)
    read_card.read(StringIO(repr(card)))
    assert read_card.name == 'k_points'
    assert read_card.subtitle == 'tpiba'
    assert read_card.value == "2\n0 0 0 0.8\n0.5 0.5 0.5 0.2"


def test_read_but_not_find(subtitled_stream, card_stream):
    from io import StringIO
    card, read_card = Card('ATOMIC_SPECIES'), Card('K_POINTS')
    read_card.read(subtitled_stream)
    card.read(card_stream)
    read_card.read(StringIO(repr(card)))
    assert read_card.name == 'k_points'
    assert read_card.subtitle == 'tpiba'
    assert read_card.value == "2\n0 0 0 0.8\n0.5 0.5 0.5 0.2"


def test_allowed_card_name_dynamic_in_constructor():
    card = Card('NeWnaMe')
    assert card.name == 'newname'


def test_card_name_must_be_in_enum():
    from traitlets import TraitError
    from pytest import raises
    card = Card('ATOMIC_SPECIES')
    with raises(TraitError):
        card.name = 'nothing'


def test_set_card_name():
    from traitlets import TraitError
    from pytest import raises
    card = Card('ATOMIC_SPECIES')
    card.name = 'k_poinTs'
    assert card.name == 'k_points'


def test_read_card(subtitled_stream, card_stream):
    from pylada.espresso.card import read_cards
    from io import StringIO
    from pylada.espresso import logger
    logger.setLevel(4)
    stream = StringIO(subtitled_stream.read() + "\n" + card_stream.read())
    result = read_cards(stream)
    assert len(result) == 2
    assert result[0].name == 'k_points'
    assert result[0].subtitle == 'tpiba'
    assert result[0].value == "2\n0 0 0 0.8\n0.5 0.5 0.5 0.2"
    assert result[1].name == 'atomic_species'
    assert result[1].subtitle is None
    assert result[1].value == 'Al 1 this.that'


def test_read_card_with_namelist(subtitled_stream, card_stream):
    from pylada.espresso.card import read_cards
    from io import StringIO
    from pylada.espresso import logger
    logger.setLevel(4)
    stream = StringIO(subtitled_stream.read() + "\n&electrons\nbullshit = 1\n/\n"
                      + "\n" + card_stream.read())
    result = read_cards(stream)
    assert len(result) == 2
    assert result[0].name == 'k_points'
    assert result[0].subtitle == 'tpiba'
    assert result[0].value == "2\n0 0 0 0.8\n0.5 0.5 0.5 0.2"
    assert result[1].name == 'atomic_species'
    assert result[1].subtitle is None
    assert result[1].value == 'Al 1 this.that'
