"""TODO To be properly documented"""

import io
from utilities.constants import atom_types
import modelcif
import modelcif.model
import modelcif.dumper


def to_modelcif(atom_positions, atom_mask, sequence):
    atom_positions = atom_positions.to('cpu').numpy()
    atom_mask = atom_mask.to('cpu').numpy()
    n = atom_positions.shape[0]
    system = modelcif.System(title='AlphaFold prediction')
    entity = modelcif.Entity(sequence, description='Model subunit')
    asym_unit = modelcif.AsymUnit(entity, details='Model subunit A', id='A')
    modeled_assembly = modelcif.Assembly([asym_unit], name='Modeled assembly')

    class _MyModel(modelcif.model.AbInitioModel):
        def get_atoms(self):
            for i in range(n):
                for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
                    if not mask:
                        continue
                    element = atom_name[0]
                    yield modelcif.model.Atom(
                        asym_unit=asym_unit,
                        type_symbol=element,
                        seq_id=i + 1,
                        atom_id=atom_name,
                        x=pos[0], y=pos[1], z=pos[2],
                        het=False,
                        occupancy=1.00
                    )

    model = _MyModel(assembly=modeled_assembly, name='Model')
    model_group = modelcif.model.ModelGroup([model], name='All models')
    system.model_groups.append(model_group)
    fh = io.StringIO()
    modelcif.dumper.write(fh, [system])

    return fh.getvalue()
