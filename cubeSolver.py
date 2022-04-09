"""A module to solve 3D cube arrangement puzzles."""
import numpy as np
from matplotlib import pyplot as plt


class Piece(object):
    """This class defines a piece of the cube."""

    def __init__(self, piece_data, color, trim=True):
        """Initialize the piece."""

        if trim:
            voxels = self.trim_to_bounding_box(piece_data)

        # determine dimensions of the piece
        self.lx = self.ly = self.lz = voxels.shape
        self.volume = voxels.sum()

        self.voxels = voxels
        self.color = color

        # generate the 24 possible orientations obtained from the 4 rotations
        # of the piece (around the z axis) about each of its 6 faces.

        # Face 1 (original orientation)
        o1 = voxels

        # Face 2, 3 & 4 (rotation around x axis)
        o2 = np.rot90(voxels, 1, (1, 2))
        o3 = np.rot90(voxels, 2, (1, 2))
        o4 = np.rot90(voxels, 3, (1, 2))

        # Faces 5 & 6 rotations around y axis
        o5 = np.rot90(voxels, -1, (0, 2))
        o6 = np.rot90(voxels, 1, (0, 2))

        base_orientations = [o1, o2, o3, o4, o5, o6]

        # TODO: recognize and remove symmetries
        self.faces = [[o, *(np.rot90(o, i) for i in [1, 2, 3])]
                             for o in base_orientations]

    def trim_to_bounding_box(self, piece_data):
        """Remove nonzero elements from around the piece_data.

        The result will be an array whose shape is reduced in dimensions
        where the original piece_data contained only nonzero elements. The flat
        representation of this new array will have nonzeros as the first and
        last element.
        """
        voxels = np.array(piece_data)
        nz = voxels.nonzero()
        f = np.array([])

        return voxels[tuple(slice(b, e + 1)
                      for b, e in zip(map(min, nz), map(max, nz)))]

    def num_orientations(self):
        return sum(len(rotations) for rotations in self.faces)

    def plot(self, flat=False):
        """Plot a 3D or 2D representation of the piece in all orientations.

        Args:
            flat: if True, plot a 2D representation of the piece.
        """
        if flat:
            data = np.array([rotation.flatten()
                             for face in self.faces
                             for rotation in face]).T
            plt.imshow(data)
            return

        fig, axs = plt.subplots(6, 4, figsize=(12, 8),
                                subplot_kw=dict(projection='3d'))
        for i, face in enumerate(self.faces):
            for ax, rotation in zip(axs[i], face):
                ax.voxels(rotation, color=self.color, edgecolor='k', alpha=0.8)
                ax.set_box_aspect(rotation.shape)
                ax.axis('off')


class Game(object):

    def __init__(self, pieces_data, colors=None, size=5):
        """Initialize the game."""
        self.size = size
        if colors is None:
            colors = []
        if len(colors) < len(pieces_data):
            n_missing = len(pieces_data) - len(colors)
            # generate the missing colors randomly
            import colorsys
            for i in np.arange(0., 360., 360. / n_missing):
                hue = i/360.
                lightness = (50 + np.random.rand() * 10)/100.
                saturation = (90 + np.random.rand() * 10)/100.
                colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        self.pieces = [Piece(piece_data, color)
                       for piece_data, color in zip(pieces_data, colors)]
        
        self.piece_volume = sum(p.volume for p in self.pieces)
        self.volume = self.size ** 3
        assert(self.piece_volume <= self.volume), "The pieces don't fit in the cube!\n" \
                                                  f"{self.piece_volume} > {self.volume}"

        self.piece_offsets = [self.num_offsets(piece) for piece in self.pieces]

        # Piece to which the game is oriented, i.e., we don't need to consider
        # it's rotations with respect to the game. To reduce the number of
        # binary variables we need to consider, we want this to be a piece with
        # a large number of offsets and a small number of symmetries (as these
        # can be used to futher reduce possibilities).
        self.base_piece_index = np.argmax(self.piece_offsets)
        
        self.piece_positions = [# range(self.num_offsets(piece))
                                # if p == self.base_piece_index
                                # else 
                                range(self.num_positions(piece))
                                for p, piece in enumerate(self.pieces)]

    def num_offsets(self, piece):
        """Find the number of possible offsets for the given piece in one orientation."""
        return int((self.size + 1 - np.array(piece.voxels.shape)).prod())

    def num_positions(self, piece):
        """Find the number of possible positions for the given piece within the game."""
        return piece.num_orientations() * self.num_offsets(piece)

    def num_possibilities(self, type=int):
        """Find all possible positions for the pieces in the game."""
        possibilities = 1
        for p, piece in enumerate(self.pieces):
            # if p == self.base_piece_index:
            #     possibilities *= self.num_offsets(piece)
            # else:
            possibilities *= self.num_positions(piece)
        possibilities = type(possibilities)
        if type == str:
            # Add space delimiters to make a more readable string

            first_break = len(possibilities) % 3
            if first_break:
                return (possibilities[0:first_break] + ' '
                        + ' '.join(possibilities[b:b+3]
                                   for b in range(first_break,
                                                  len(possibilities), 3)))
            else:
                return (' '.join(possibilities[b:b+3]
                                 for b in range(first_break,
                                                len(possibilities), 3)))
        return possibilities

    def num_binaries(self):
        """Find the number of binary variables representations of the game."""
        binaries = 0
        for p, piece in enumerate(self.pieces):
            # if p == self.base_piece_index:
            #     binaries += self.num_offsets(piece)
            # else:
            binaries += self.num_positions(piece)
        binaries
        return binaries

    def yield_offsets(self, p, f, r):
        """Find the offsets for the piece with the given number face and rotation."""
        from itertools import product
        max_offsets = self.size + 1 - np.array(self.pieces[p].faces[f][r].shape)
        yield from product(*map(range, max_offsets))

    def plot_pieces(self):
        """Plot the piece with the given number."""
        # Determine a gridsize for plotting the pieces
        gridsize = int(np.ceil(len(self.pieces) ** 0.5))

        if gridsize == 1:
            fig, ax = plt.subplots()
            ax.voxels(self.pieces[0].voxels, color=self.pieces[0].color, alpha=0.8)
            ax.set_box_aspect(piece.voxels.shape)
            ax.axis('off')
            plt.draw()
            return

        fig, axs = plt.subplots(gridsize, gridsize, figsize=(8, 8),
                                subplot_kw=dict(projection='3d'))
        # Plot the pieces
        for i, piece in enumerate(self.pieces):
            ax = axs[i // gridsize, i % gridsize]
            ax.voxels(piece.voxels, color=piece.color, edgecolor='k', alpha=0.8)
            ax.set_box_aspect(piece.voxels.shape)
            ax.axis('off')

        # Turn off remaining axes
        for j in range(i, gridsize ** 2):
            ax = axs[j // gridsize, j % gridsize]
            ax.axis('off')
        plt.draw()

    def solve(self, solver='gurobi', executable=None, plot=True, **kwargs):
        """Solve the problem using an MILP solver.
        
        Args
        ----
        solver: str
            The solver to use, some options are:
            'cbc', 'glpk', 'scip', 'cplex', 'gurobi'
        executable: str
            The path to the solver executable. If None is given,
            the solver is assumed to be in the PATH. If it is not
            found, an attempt is made to dispatch soltion to the
            NEOS server.
        kwargs: dict
            Additional arguments to pass to the solver.
        
        Returns
        -------
        solution: np.ndarray
            The solution to the problem. It has the same shape as
            the game and contains the piece id occupying each voxel.
        """
        # Compute the competition for positions (sum over piece positions
        # covering a particular voxel) and the piece positions coverint a
        # particular voxel.
        competition = np.empty(self.size**3)
        coverings = [[] for _ in range(self.size**3)]
        for p, piece in enumerate(self.pieces):
            num_o = self.piece_offsets[p]
            # if p == self.base_piece_index:
            #     rotation = piece.faces[0][0]
            #     nzr = np.array(np.nonzero(rotation)).T
            #     for o, offset in enumerate(self.yield_offsets(0, 0, 0)):
            #         for x, y, z in nzr + offset:
            #               v = x * self.size**2 + y * self.size + z
            #               competition[v] += 1
            #               coverings[v].append((p, o))
            #               # covers[x][y][z].append((0, 0, 0, o))
            #     continue
            for f, face in enumerate(piece.faces):
                for r, rotation in enumerate(face):
                    nzr = np.array(np.nonzero(rotation)).T
                    for o, offset in enumerate(self.yield_offsets(p, f, r)):
                        pos = f * 4 * num_o + r * num_o + o
                        for x, y, z in nzr + offset:
                            v = x * self.size**2 + y * self.size + z
                            competition[v] += 1
                            coverings[v].append((p, pos))
                            # covers[x][y][z].append((p, f, r, o))

        from pyomo.environ import (ConcreteModel, Set, Var, Binary, Constraint, Objective, SolverFactory)

        m = ConcreteModel()
        m.pieces = Set(initialize=range(len(self.pieces)), name='pieces')
        m.positions = Set(m.pieces, name="positions",
                          initialize=lambda m, p: self.piece_positions[p])

        def PP_rule(m):
            return [(p, pos) for p in m.pieces for pos in m.positions[p]]

        m.piece_positions = Set(initialize=PP_rule)

        m.piece_position_choice = Var(m.piece_positions, domain=Binary,
                                      bounds=(0, 1), name='b')

        # Each piece can only be in one position
        def piece_is_placed_rule(m, p):
            return sum(m.piece_position_choice[p, pos]
                       for pos in m.positions[p]) == 1

        m.piece_is_placed = Constraint(m.pieces, rule=piece_is_placed_rule)

        m.voxels = Set(initialize=range(self.size**3), name='voxels')

        # m.slack = Var(m.voxels, bounds=(0, 1), name='slack')

        # Each voxel can only be covered by one piece at most
        def voxel_is_covered_rule(m, v):
            return sum(m.piece_position_choice[p, pos]
                    for p, pos in coverings[v]) <= 1
                    #    for p, pos in coverings[v]) + m.slack[v] == 1

        m.voxel_is_covered = Constraint(m.voxels, rule=voxel_is_covered_rule)

        # Minimize competition
        # m.obj = Objective(expr=sum(coeff * m.piece_position_choice[p, pos]
        #                            for v, coeff in enumerate(competition)
        #                            for p, pos in coverings[v]) - sum(competition))

        # Minimize the active piece positions
        m.obj = Objective(expr=sum(m.piece_position_choice[p, pos]
                                   for p, pos in m.piece_positions))

        # Minimize the slack
        # m.obj = Objective(expr=sum(m.slack[v] for v in m.voxels))

        import shutil
        if executable is None:
            executable = shutil.which(solver)
            if executable is None:
                import os
                prompt = f"No executable found locally for solver '{solver}' " \
                          "Trying to dispatch to NEOS server instead, please " \
                          "enter your email address:\n"
                os.environ['NEOS_EMAIL'] = input(prompt)
                from pyomo.environ import SolverManagerFactory
                sm = SolverManagerFactory('neos')
                results = sm.solve(m, opt=solver, **kwargs)

        if executable is not None:
            if solver.lower() == 'gurobi':
                opt = SolverFactory(solver, executable=executable, solver_io="python")
            else:
                opt = SolverFactory(solver, executable=executable)
            results = opt.solve(m, **kwargs)

        # Solve using cbc
        # opt = SolverFactory("cbc", executable="/usr/local/bin/cbc", solver_io="nl")

        # Solve using gurobi
        # opt = SolverFactory("gurobi", solver_io="python")

        tol = 1e-4
        solution = np.full((self.size, self.size, self.size), float('nan'))

        for p, pos in m.piece_position_choice.keys():
            if abs(m.piece_position_choice[p, pos].value) > tol:
                # print(p, pos, m.piece_position_choice[p, pos].value)  
                num_o = self.piece_offsets[p]
                # if p == self.base_piece_index:
                #     f, r, o = np.unravel_index(pos, (1, 1, num_o))
                # else:
                f, r, o = np.unravel_index(pos, (6, 4, num_o))
                # print(p, pos, '(', f, r, o, ')',
                #       m.piece_position_choice[p, pos].value)
                offset = [*self.yield_offsets(p, f, r)][o]

                position = self.pieces[p].faces[f][r]
                voxels = tuple((np.array(position.nonzero()).T + offset).T)
                solution[voxels] = p

        if plot:
            import matplotlib
            c = np.empty_like(solution, dtype='O')
            for p, piece in enumerate(self.pieces):
                for i in np.array(np.where(solution == p)).T:
                    c[tuple(i)] = matplotlib.colors.to_hex(piece.color)

            gridsize = int(np.ceil(np.sqrt(self.size)))
            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(gridsize, 2 * gridsize)

            for p, piece in enumerate(self.pieces):
                ax = fig.add_subplot(gs[p // gridsize, p % gridsize], projection='3d')
                ax.voxels(solution == p, color=piece.color, edgecolor='k', alpha=0.8)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.set_xticks(range(self.size))
                ax.set_yticks(range(self.size))
                ax.set_zticks(range(self.size))
            ax = fig.add_subplot(gs[:, gridsize+1:], projection='3d')
            # Note that the voxels method requires a 3D array of bools
            # Hence, we add 1 to the solution array to ensure the first
            # piece is plotted as well.
            ax.voxels(solution + 1, facecolors=c, edgecolor='k')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xticks(range(self.size))
            ax.set_yticks(range(self.size))
            ax.set_zticks(range(self.size))
            plt.draw()
        return results, solution


if __name__ == '__main__':

    import sys

    try:
        solver = sys.argv[1]
    except IndexError:
        solver = 'gurobi'

    from example import pieces

    self = Game(pieces)
    # piece = self.pieces[0]
    # piece.plot()
    self.plot_pieces()
    solution = self.solve(solver, tee=True)
    plt.show()