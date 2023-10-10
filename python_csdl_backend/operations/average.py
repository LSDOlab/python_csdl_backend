from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
from python_csdl_backend.utils.sparse_utils import get_sparsity
from python_csdl_backend.operations.max import ScalarExtremumLite
import numpy as np
import scipy.sparse as sp


def get_average_lite(op):
    # raise NotImplementedError('average not yet implemented')
    if len(op.dependencies) == 1 and op.literals['axes'] != None:
        return SingleTensorAverageLite
    elif len(op.dependencies) == 1 and op.literals['axes'] == None:
        return SingleTensorAverageLite
    else:
        return MultipleTensorAverageLite


class SingleTensorAverageLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        name = 'average'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.in_name = operation.dependencies[0].name
        self.shape = operation.dependencies[0].shape
        self.out_name = operation.outs[0].name
        self.out_shape = operation.outs[0].shape
        self.axes = operation.literals['axes']
        # self.val = operation.dependencies[0].val

        in_name = self.in_name
        shape = self.shape
        out_name = self.out_name
        out_shape = self.out_shape
        # print(out_shape)
        axes = self.axes
        # val = self.val

        # Computation of Output shape if the shape is not provided
        if out_shape != None:
            self.output_shape = out_shape
        elif axes != None:
            output_shape = np.delete(shape, axes)
            self.output_shape = tuple(output_shape)

        input_size = np.prod(shape)
        self.input_size = input_size
        self.output_size = np.prod(self.output_shape)

        # axes == None works for any tensor
        # self.add_input(in_name, shape=shape, val=val)
        if axes == None:
            self.val = np.full((input_size, ), 1. / input_size)
            self.nnz = len(self.val.flatten())

        # axes != None works only for matrices
        else:
            # self.add_output(out_name, shape=self.output_shape)
            self.cols = np.arange(input_size)

            rows = np.unravel_index(np.arange(input_size), shape=shape)
            rows = np.delete(np.array(rows), axes, axis=0)
            self.rows = np.ravel_multi_index(rows, dims=self.output_shape)

            num_entries_averaged = np.prod(np.array(shape)[axes])
            self.val = np.full((input_size, ), 1. / num_entries_averaged)
            self.nnz = len(self.val.flatten())

    def get_evaluation(self, eval_block, vars):
        in_name = self.in_name
        out_name = self.out_name
        axes = self.axes

        # axes == None works for any tensor
        if axes == None:
            eval_block.write(f'{self.get_output_id(out_name)} = np.array(np.average({self.get_input_id(in_name)})).reshape((1,))')
            # eval_block.write(f'{self.get_output_id(out_name)} = np.average({self.get_input_id(in_name)})')

        # axes != None works only for matrices
        else:
            axis_name = f'{self.get_input_id(in_name)}_{self.get_output_id(out_name)}_axes'
            vars[axis_name] = axes
            eval_block.write(f'{self.get_output_id(out_name)} = np.average({self.get_input_id(in_name)}, axis={axis_name})')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):

        key_tuple = get_only(partials_dict)
        input = key_tuple[1].id
        output = key_tuple[0].id
        partial_name = partials_dict[key_tuple]['name']

        if self.axes == None:
            if is_sparse_jac:
                vars[partial_name] = sp.csc_matrix(self.val)
            else:
                vars[partial_name] = self.val.reshape((1,self.input_size))
        else:
            if is_sparse_jac:
                vars[partial_name] = sp.csc_matrix((self.val.flatten(), (self.rows, self.cols)), shape=(self.output_size, self.input_size))
            else:
                vars[partial_name] = np.zeros((self.output_size, self.input_size))
                vars[partial_name][self.rows, self.cols] = self.val.flatten()

    def determine_sparse(self):
        if self.input_size < 100:
            return False

        if get_sparsity(self.nnz, self.output_size, self.input_size) < SPARSE_SIZE_CUTOFF:
            return True
        return False


class MultipleTensorAverageLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name, **kwargs):
        name = 'average_multiple'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        in_names = [var.name for var in operation.dependencies]
        shape = operation.dependencies[0].shape
        out_name = operation.outs[0].name
        out_shape = operation.outs[0].shape
        axes = operation.literals['axes']
        self.in_names = in_names
        self.out_name = out_name
        self.axes = axes

        # vals = [var.val for var in operation.dependencies]
        if out_shape == None and axes != None:
            output_shape = np.delete(shape, axes)
            self.output_shape = tuple(output_shape)
        else:
            self.output_shape = out_shape

        self.num_inputs = len(in_names)
        input_size = np.prod(shape)
        self.input_size = input_size
        self.output_size = np.prod(self.output_shape)

        # axes not specified => elementwise average
        if axes == None:
            # self.add_output(out_name, shape=shape)

            self.val = np.full((input_size, ), 1. / self.num_inputs)
            self.rows = self.cols = np.arange(input_size)

            # for in_name, in_val in zip(in_names, vals):
                # self.add_input(in_name, shape=shape, val=in_val)
                # self.declare_partials(out_name,
                #                       in_name,
                #                       rows=rows,
                #                       cols=cols,
                #                       val=val)

        # axes specified => axiswise average
        else:
            # self.add_output(out_name, shape=self.output_shape)
            self.cols = np.arange(input_size)

            rows = np.unravel_index(np.arange(input_size), shape=shape)
            rows = np.delete(np.array(rows), axes, axis=0)
            self.rows = np.ravel_multi_index(rows, dims=self.output_shape)

            num_entries_averaged = np.prod(np.array(shape)[axes])
            self.val = np.full((input_size, ),
                               1. / (num_entries_averaged * self.num_inputs))
            # for in_name, in_val in zip(in_names, vals):
            #     self.add_input(in_name, shape=shape, val=in_val)
            #     self.declare_partials(out_name,
            #                           in_name,
            #                           rows=rows,
            #                           cols=cols,
            #                           val=val)
        self.nnz = len(self.rows)

    def get_evaluation(self, eval_block, vars):
        in_names = self.in_names
        out_name = self.out_name
        axes = self.axes
        out_shape = self.output_shape

        # axes == None does the elementwise average of the tensors
        if axes == None:
            # outputs[out_name] = inputs[in_names[0]]
            # for i in range(1, self.num_inputs):
            #     outputs[out_name] += inputs[in_names[i]]
            # outputs[out_name] = outputs[out_name] / self.num_inputs

            out_id = self.get_output_id(out_name)
            eval_block.write(f'{out_id} = np.zeros({out_shape})')
            for i in range(0, self.num_inputs):
                eval_block.write(f'{out_id} += {self.get_input_id(in_names[i])}')
            eval_block.write(f'{out_id} = {out_id}/{self.num_inputs}')

        # axes != None takes the average along specified axes
        else:
            # outputs[out_name] = np.average(inputs[in_names[0]], axis=axes)
            # for i in range(1, self.num_inputs):
            #     outputs[out_name] += np.average(inputs[in_names[i]], axis=axes)
            # outputs[out_name] = outputs[out_name] / self.num_inputs

            out_id = self.get_output_id(out_name)
            eval_block.write(f'{out_id} = np.average({self.get_input_id(in_names[0])}, axis={axes})')
            for i in range(1, self.num_inputs):
                eval_block.write(f'{out_id} += np.average({self.get_input_id(in_names[i])}, axis={axes})')
            eval_block.write(f'{out_id} = {out_id}/{self.num_inputs}')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):
        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            if is_sparse_jac:
                vars[partial_name] = sp.csc_matrix((self.val.flatten(), (self.rows, self.cols)), shape=(self.output_size, self.input_size))
            else:
                vars[partial_name] = np.zeros((self.output_size, self.input_size))
                vars[partial_name][self.rows, self.cols] = self.val.flatten()

    def determine_sparse(self):
        if self.input_size < 100:
            return False

        if get_sparsity(self.nnz, self.output_size, self.input_size) < SPARSE_SIZE_CUTOFF:
            return True
        return False