from python_csdl_backend.operations.operation_base import OperationBase
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import to_unique_list, get_scalars_list
from python_csdl_backend.utils.general_utils import get_only
from python_csdl_backend.utils.operation_utils import SPARSE_SIZE_CUTOFF
import numpy as np
import scipy.sparse as sp


def get_quatrotvec_lite(op):
    return QuatrotvecLite


class QuatrotvecLite(OperationBase):

    def __init__(self, operation, nx_inputs, nx_outputs, name='', **kwargs):
        op_name = 'quatrotvec'
        name = f'{name}_{op_name}'
        super().__init__(operation, nx_inputs, nx_outputs, name, **kwargs)

        self.shape = operation.dependencies[1].shape
        self.quat_name = operation.dependencies[0].name
        self.quat_size = np.prod(operation.dependencies[0].shape)
        self.vec_name = operation.dependencies[1].name
        self.vec_size = np.prod(operation.dependencies[1].shape)
        # self.quat_vals = operation.dependencies[0].val
        # self.vec_vals = operation.dependencies[1].val
        self.out_name = operation.outs[0].name
        self.out_id = self.get_output_id(self.out_name)
        self.out_size = np.prod(operation.outs[0].shape)
        shape = self.shape
        shape = shape[:-1]
        size = np.prod(shape)
        # out_indices = np.arange(size*3).reshape(shape + (3,))
        # vec_indices = np.arange(size*3).reshape(shape + (3,))
        # quat_indices = np.arange(size*4).reshape(shape + (4,))

        # r0 = np.zeros(shape + (3, 3))
        # c0 = np.zeros(shape + (3, 3))

        # for i in range(3):
        #     for j in range(3):
        #         r0[..., i, j] = out_indices[..., i]
        #         c0[..., i, j] = vec_indices[..., j]

        # self.r0 = r0.flatten()
        # self.c0 = c0.flatten()

        # r1 = np.zeros(shape + (3, 4))
        # c1 = np.zeros(shape + (3, 4))

        # for i in range(3):
        #     for j in range(4):
        #         r1[..., i, j] = out_indices[..., i]
        #         c1[..., i, j] = quat_indices[..., j]

        # self.r1 = r1.flatten()
        # self.c1 = c1.flatten()
        self.quat_id = self.get_input_id(self.quat_name)
        self.vec_id = self.get_input_id(self.vec_name)

    def dot_quat_vec(self, quat, vec):
        dot_quat_vec = vec[..., 0]*quat[..., 1] + vec[..., 1]*quat[..., 2] + vec[..., 2]*quat[..., 3]
        return dot_quat_vec

    def dot_quat_quat(self, quat):
        dot_quat_quat = quat[..., 1]**2 + quat[..., 2]**2 + quat[..., 3]**2
        return dot_quat_quat

    def execute(self, quat, vec):
        shape = self.shape
        shape = shape[:-1]
        temp = np.zeros(shape + (3,))
        temp[..., 0] = 2 * self.dot_quat_vec(quat, vec) * quat[..., 1] + (quat[..., 0]**2 - self.dot_quat_quat(quat)) * \
            vec[..., 0] + 2*quat[..., 0] * (quat[..., 2]*vec[..., 2] - quat[..., 3]*vec[..., 1])
        temp[..., 1] = 2 * self.dot_quat_vec(quat, vec) * quat[..., 2] + (quat[..., 0]**2 - self.dot_quat_quat(quat)) * \
            vec[..., 1] + 2*quat[..., 0] * (quat[..., 3]*vec[..., 0] - quat[..., 1]*vec[..., 2])
        temp[..., 2] = 2 * self.dot_quat_vec(quat, vec) * quat[..., 3] + (quat[..., 0]**2 - self.dot_quat_quat(quat)) * \
            vec[..., 2] + 2*quat[..., 0] * (quat[..., 1]*vec[..., 1] - quat[..., 2]*vec[..., 0])
        return temp.copy()

    def get_evaluation(self, eval_block, vars):

        executable_name = self.name+'_exec'
        vars[executable_name] = self.execute

        str = f'{self.out_id} = {executable_name}({self.quat_id}, {self.vec_id})'
        eval_block.write(str)

    def partials_func(self, vec,  quat):

        shape = self.shape
        shape = shape[:-1]
        temp = np.zeros(shape + (3, 3))

        temp[..., 0, 0] = 2*quat[..., 1]**2 + quat[..., 0]**2 - quat[..., 1]**2 - quat[..., 2]**2 - quat[..., 3]**2
        temp[..., 0, 1] = 2*quat[..., 2]*quat[..., 1] - 2*quat[..., 0]*quat[..., 3]
        temp[..., 0, 2] = 2*quat[..., 3]*quat[..., 1] + 2*quat[..., 0]*quat[..., 2]

        temp[..., 1, 0] = 2*quat[..., 1]*quat[..., 2] + 2*quat[..., 0]*quat[..., 3]
        temp[..., 1, 1] = 2*quat[..., 2]**2 + quat[..., 0]**2 - quat[..., 1]**2 - quat[..., 2]**2 - quat[..., 3]**2
        temp[..., 1, 2] = 2*quat[..., 3]*quat[..., 2] - 2*quat[..., 0]*quat[..., 1]

        temp[..., 2, 0] = 2*quat[..., 1]*quat[..., 3] - 2*quat[..., 0]*quat[..., 2]
        temp[..., 2, 1] = 2*quat[..., 2]*quat[..., 3] + 2*quat[..., 0]*quat[..., 1]
        temp[..., 2, 2] = 2*quat[..., 3]**2 + quat[..., 0]**2 - quat[..., 1]**2 - quat[..., 2]**2 - quat[..., 3]**2

        temp1 = np.zeros(shape + (3, 4))

        temp1[..., 0, 0] = 2*quat[..., 0]*vec[..., 0] + 2*quat[..., 2]*vec[..., 2] - 2*quat[..., 3]*vec[..., 1]
        temp1[..., 0, 1] = 4*vec[..., 0]*quat[..., 1] + 2*vec[..., 1]*quat[..., 2] + 2*vec[..., 2]*quat[..., 3] - 2*vec[..., 0]*quat[..., 1]
        temp1[..., 0, 2] = 2*vec[..., 1]*quat[..., 1] - 2*quat[..., 2]*vec[..., 0] + 2*quat[..., 0]*vec[..., 2]
        temp1[..., 0, 3] = 2*vec[..., 2]*quat[..., 1] - 2*quat[..., 3]*vec[..., 0] - 2*quat[..., 0]*vec[..., 1]

        temp1[..., 1, 0] = 2*quat[..., 0]*vec[..., 1] + 2*quat[..., 3]*vec[..., 0] - 2*quat[..., 1]*vec[..., 2]
        temp1[..., 1, 1] = 2*vec[..., 0]*quat[..., 2] - 2*quat[..., 1]*vec[..., 1] - 2*quat[..., 0]*vec[..., 2]
        temp1[..., 1, 2] = 4*vec[..., 1]*quat[..., 2] + 2*vec[..., 0]*quat[..., 1] + 2*vec[..., 2]*quat[..., 3] - 2*quat[..., 2]*vec[..., 1]
        temp1[..., 1, 3] = 2*vec[..., 2]*quat[..., 2] - 2*quat[..., 3]*vec[..., 1] + 2*quat[..., 0]*vec[..., 0]

        temp1[..., 2, 0] = 2*quat[..., 0]*vec[..., 2] + 2*quat[..., 1]*vec[..., 1] - 2*quat[..., 2]*vec[..., 0]
        temp1[..., 2, 1] = 2*vec[..., 0]*quat[..., 3] - 2*quat[..., 1]*vec[..., 2] + 2*quat[..., 0]*vec[..., 1]
        temp1[..., 2, 2] = 2*vec[..., 1]*quat[..., 3] - 2*quat[..., 2]*vec[..., 2] - 2*quat[..., 0]*vec[..., 0]
        temp1[..., 2, 3] = 2*vec[..., 0]*quat[..., 1] + 2*vec[..., 1]*quat[..., 2] + 4*vec[..., 2]*quat[..., 3] - 2*quat[..., 3]*vec[..., 2]

        popv = temp.flatten()
        popq = temp1.flatten()

        return popv, popq

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):
        shape = self.shape
        shape = shape[:-1]
        size = np.prod(shape)

        partials_func_name = self.name+'_partials_func'
        out_tuple_name = self.name+'_partials_out'

        vars[partials_func_name] = self.partials_func
        partials_block.write(f'{out_tuple_name} = {partials_func_name}({self.vec_id},{self.quat_id})')
        if lazy:

            partials_block.write(f'out_indices = np.arange({size}*3).reshape({shape} + (3,))')
            partials_block.write(f'vec_indices = np.arange({size}*3).reshape({shape} + (3,))')
            partials_block.write(f'quat_indices = np.arange({size}*4).reshape({shape} + (4,))')
        else:
            out_indices = np.arange(size*3).reshape(shape + (3,))
            vec_indices = np.arange(size*3).reshape(shape + (3,))
            quat_indices = np.arange(size*4).reshape(shape + (4,))

            r0 = np.zeros(shape + (3, 3))
            c0 = np.zeros(shape + (3, 3))

            for i in range(3):
                for j in range(3):
                    r0[..., i, j] = out_indices[..., i]
                    c0[..., i, j] = vec_indices[..., j]

            self.r0 = r0.flatten()
            self.c0 = c0.flatten()

            r1 = np.zeros(shape + (3, 4))
            c1 = np.zeros(shape + (3, 4))

            for i in range(3):
                for j in range(4):
                    r1[..., i, j] = out_indices[..., i]
                    c1[..., i, j] = quat_indices[..., j]

            self.r1 = r1.flatten()
            self.c1 = c1.flatten()


        for key_tuple in partials_dict:
            input = key_tuple[1].id
            output = key_tuple[0].id
            partial_name = partials_dict[key_tuple]['name']

            if not lazy:
                row_name = self.name+input+'_rows'
                col_name = self.name+input+'_cols'
                if input == self.vec_id:
                    index = 0
                    vars[row_name] = self.r0.astype(int)
                    vars[col_name] = self.c0.astype(int)
                    in_size = self.vec_size
                else:
                    index = 1
                    vars[row_name] = self.r1.astype(int)
                    vars[col_name] = self.c1.astype(int)
                    in_size = self.quat_size

                partials_block.write(f'vals = {out_tuple_name}[{index}]')
                if is_sparse_jac:
                    partials_block.write(f'{partial_name} = sp.csc_matrix((vals, ({row_name},{col_name})), shape = ({self.out_size},{in_size}))')
                else:
                    partials_block.write(f'{partial_name} = np.zeros(({self.out_size},{in_size}))')
                    partials_block.write(f'{partial_name}[{row_name}, {col_name}] = vals')

            else:
        
                if input == self.vec_id:
                    index = 0
                    in_size = self.vec_size

                    partials_block.write(f'r0 = np.zeros({shape} + (3, 3))')
                    partials_block.write(f'c0 = np.zeros({shape} + (3, 3))')
                    
                    partials_block.write(f"""
for i in range(3):
    for j in range(3):
        r0[..., i, j] = out_indices[..., i]
        c0[..., i, j] = vec_indices[..., j]
""")
                    partials_block.write(f'rows = r0.flatten().astype(int)')
                    partials_block.write(f'cols = c0.flatten().astype(int)')

                else:
                    index = 1
                    in_size = self.quat_size

                    partials_block.write(f'r1 = np.zeros({shape} + (3, 4))')
                    partials_block.write(f'c1 = np.zeros({shape} + (3, 4))')
                    
                    partials_block.write(f"""
for i in range(3):
    for j in range(4):
        r1[..., i, j] = out_indices[..., i]
        c1[..., i, j] = quat_indices[..., j]
""")
                    partials_block.write(f'rows = r1.flatten().astype(int)')
                    partials_block.write(f'cols = c1.flatten().astype(int)')

                partials_block.write(f'vals = {out_tuple_name}[{index}]')
                if is_sparse_jac:
                    partials_block.write(f'{partial_name} = sp.csc_matrix((vals, (rows, cols)), shape = ({self.out_size},{in_size}))')
                else:
                    partials_block.write(f'{partial_name} = np.zeros(({self.out_size},{in_size}))')
                    partials_block.write(f'{partial_name}[rows, cols] = vals')
                partials_block.write(f'del rows')
                partials_block.write(f'del cols')
                partials_block.write(f'del vals')
        if lazy:
            partials_block.write(f'del out_indices')
            partials_block.write(f'del vec_indices')
            partials_block.write(f'del quat_indices')
    def determine_sparse(self):
        return True
