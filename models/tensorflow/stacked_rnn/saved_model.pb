ٰ
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48ݻ
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
,rnn/stacked_rnn_cells/simple_rnn_cell_2/biasVarHandleOp*
_output_shapes
: *=

debug_name/-rnn/stacked_rnn_cells/simple_rnn_cell_2/bias/*
dtype0*
shape: *=
shared_name.,rnn/stacked_rnn_cells/simple_rnn_cell_2/bias
�
@rnn/stacked_rnn_cells/simple_rnn_cell_2/bias/Read/ReadVariableOpReadVariableOp,rnn/stacked_rnn_cells/simple_rnn_cell_2/bias*
_output_shapes
: *
dtype0
�
8rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *I

debug_name;9rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel/*
dtype0*
shape
:  *I
shared_name:8rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel
�
Lrnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp8rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel*
_output_shapes

:  *
dtype0
�
.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernelVarHandleOp*
_output_shapes
: *?

debug_name1/rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel/*
dtype0*
shape
:  *?
shared_name0.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel
�
Brnn/stacked_rnn_cells/simple_rnn_cell_2/kernel/Read/ReadVariableOpReadVariableOp.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel*
_output_shapes

:  *
dtype0
�
,rnn/stacked_rnn_cells/simple_rnn_cell_1/biasVarHandleOp*
_output_shapes
: *=

debug_name/-rnn/stacked_rnn_cells/simple_rnn_cell_1/bias/*
dtype0*
shape: *=
shared_name.,rnn/stacked_rnn_cells/simple_rnn_cell_1/bias
�
@rnn/stacked_rnn_cells/simple_rnn_cell_1/bias/Read/ReadVariableOpReadVariableOp,rnn/stacked_rnn_cells/simple_rnn_cell_1/bias*
_output_shapes
: *
dtype0
�
8rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *I

debug_name;9rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel/*
dtype0*
shape
:  *I
shared_name:8rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel
�
Lrnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp8rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel*
_output_shapes

:  *
dtype0
�
.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernelVarHandleOp*
_output_shapes
: *?

debug_name1/rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel/*
dtype0*
shape
:  *?
shared_name0.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel
�
Brnn/stacked_rnn_cells/simple_rnn_cell_1/kernel/Read/ReadVariableOpReadVariableOp.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel*
_output_shapes

:  *
dtype0
�
*rnn/stacked_rnn_cells/simple_rnn_cell/biasVarHandleOp*
_output_shapes
: *;

debug_name-+rnn/stacked_rnn_cells/simple_rnn_cell/bias/*
dtype0*
shape: *;
shared_name,*rnn/stacked_rnn_cells/simple_rnn_cell/bias
�
>rnn/stacked_rnn_cells/simple_rnn_cell/bias/Read/ReadVariableOpReadVariableOp*rnn/stacked_rnn_cells/simple_rnn_cell/bias*
_output_shapes
: *
dtype0
�
6rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *G

debug_name97rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel/*
dtype0*
shape
:  *G
shared_name86rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel
�
Jrnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp6rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel*
_output_shapes

:  *
dtype0
�
,rnn/stacked_rnn_cells/simple_rnn_cell/kernelVarHandleOp*
_output_shapes
: *=

debug_name/-rnn/stacked_rnn_cells/simple_rnn_cell/kernel/*
dtype0*
shape
: *=
shared_name.,rnn/stacked_rnn_cells/simple_rnn_cell/kernel
�
@rnn/stacked_rnn_cells/simple_rnn_cell/kernel/Read/ReadVariableOpReadVariableOp,rnn/stacked_rnn_cells/simple_rnn_cell/kernel*
_output_shapes

: *
dtype0
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape
: @*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: @*
dtype0
�
serving_default_input_1Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1,rnn/stacked_rnn_cells/simple_rnn_cell/kernel*rnn/stacked_rnn_cells/simple_rnn_cell/bias6rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel,rnn/stacked_rnn_cells/simple_rnn_cell_1/bias8rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel,rnn/stacked_rnn_cells/simple_rnn_cell_2/bias8rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_73954

NoOpNoOp
�B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�A
value�AB�A B�A
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias*
r
40
51
62
73
84
95
:6
;7
<8
"9
#10
*11
+12
213
314*
r
40
51
62
73
84
95
:6
;7
<8
"9
#10
*11
+12
213
314*
* 
�
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Btrace_0
Ctrace_1* 

Dtrace_0
Etrace_1* 
* 

Fserving_default* 
C
40
51
62
73
84
95
:6
;7
<8*
C
40
51
62
73
84
95
:6
;7
<8*
* 
�

Gstates
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Mtrace_0
Ntrace_1
Otrace_2
Ptrace_3* 
6
Qtrace_0
Rtrace_1
Strace_2
Ttrace_3* 
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
	[cells*
* 
* 
* 
* 
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

atrace_0* 

btrace_0* 

"0
#1*

"0
#1*
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

htrace_0* 

itrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

20
31*

20
31*
* 
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

vtrace_0* 

wtrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,rnn/stacked_rnn_cells/simple_rnn_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*rnn/stacked_rnn_cells/simple_rnn_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,rnn/stacked_rnn_cells/simple_rnn_cell_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,rnn/stacked_rnn_cells/simple_rnn_cell_2/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

x0*
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
C
40
51
62
73
84
95
:6
;7
<8*
C
40
51
62
73
84
95
:6
;7
<8*
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

~trace_0
trace_1* 

�trace_0
�trace_1* 

�0
�1
�2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
* 

�0
�1
�2*
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

4kernel
5recurrent_kernel
6bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

7kernel
8recurrent_kernel
9bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

:kernel
;recurrent_kernel
<bias*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

40
51
62*

40
51
62*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

70
81
92*

70
81
92*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

:0
;1
<2*

:0
;1
<2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias,rnn/stacked_rnn_cells/simple_rnn_cell/kernel6rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel*rnn/stacked_rnn_cells/simple_rnn_cell/bias.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel8rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel,rnn/stacked_rnn_cells/simple_rnn_cell_1/bias.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel8rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel,rnn/stacked_rnn_cells/simple_rnn_cell_2/biastotalcountConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_75119
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias,rnn/stacked_rnn_cells/simple_rnn_cell/kernel6rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel*rnn/stacked_rnn_cells/simple_rnn_cell/bias.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel8rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel,rnn/stacked_rnn_cells/simple_rnn_cell_1/bias.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel8rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel,rnn/stacked_rnn_cells/simple_rnn_cell_2/biastotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_75179��
�
�
1__inference_stacked_rnn_cells_layer_call_fn_74877

inputs
states_0
states_1
states_2
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5:  
	unknown_6: 
	unknown_7:  
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1states_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:��������� :��������� :��������� :��������� *+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_72850o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:��������� q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:���������:��������� :��������� :��������� : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_1:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_2:%!

_user_specified_name74851:%!

_user_specified_name74853:%!

_user_specified_name74855:%!

_user_specified_name74857:%!

_user_specified_name74859:%	!

_user_specified_name74861:%
!

_user_specified_name74863:%!

_user_specified_name74865:%!

_user_specified_name74867
�x
�
>__inference_rnn_layer_call_and_return_conditional_losses_74410
inputs_0R
@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: O
Astacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: T
Bstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity��8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
(stacked_rnn_cells/simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd2stacked_rnn_cells/simple_rnn_cell/MatMul:product:0@stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulzeros:output:0Astacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%stacked_rnn_cells/simple_rnn_cell/addAddV22stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:04stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&stacked_rnn_cells/simple_rnn_cell/ReluRelu)stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul4stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulzeros_1:output:0Cstacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_1/addAddV24stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_1/ReluRelu+stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul6stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulzeros_2:output:0Cstacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_2/addAddV24stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_2/ReluRelu+stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*k
_output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *+
_read_only_resource_inputs
		
*
bodyR
while_body_74301*
condR
while_cond_74300*j
output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp9^stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8^stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:������������������: : : : : : : : : 2t
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_74785

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:��������� X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�x
�
>__inference_rnn_layer_call_and_return_conditional_losses_74228
inputs_0R
@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: O
Astacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: T
Bstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity��8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
(stacked_rnn_cells/simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd2stacked_rnn_cells/simple_rnn_cell/MatMul:product:0@stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulzeros:output:0Astacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%stacked_rnn_cells/simple_rnn_cell/addAddV22stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:04stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&stacked_rnn_cells/simple_rnn_cell/ReluRelu)stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul4stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulzeros_1:output:0Cstacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_1/addAddV24stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_1/ReluRelu+stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul6stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulzeros_2:output:0Cstacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_2/addAddV24stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_2/ReluRelu+stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*k
_output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *+
_read_only_resource_inputs
		
*
bodyR
while_body_74119*
condR
while_cond_74118*j
output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp9^stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8^stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:������������������: : : : : : : : : 2t
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource
�:
�	
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_73053

inputs

states
states_1
states_2@
.simple_rnn_cell_matmul_readvariableop_resource: =
/simple_rnn_cell_biasadd_readvariableop_resource: B
0simple_rnn_cell_matmul_1_readvariableop_resource:  B
0simple_rnn_cell_1_matmul_readvariableop_resource:  ?
1simple_rnn_cell_1_biasadd_readvariableop_resource: D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:  B
0simple_rnn_cell_2_matmul_readvariableop_resource:  ?
1simple_rnn_cell_2_biasadd_readvariableop_resource: D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity

identity_1

identity_2

identity_3��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�(simple_rnn_cell_1/BiasAdd/ReadVariableOp�'simple_rnn_cell_1/MatMul/ReadVariableOp�)simple_rnn_cell_1/MatMul_1/ReadVariableOp�(simple_rnn_cell_2/BiasAdd/ReadVariableOp�'simple_rnn_cell_2/MatMul/ReadVariableOp�)simple_rnn_cell_2/MatMul_1/ReadVariableOp�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
simple_rnn_cell/MatMulMatMulinputs-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell/MatMul_1MatMulstates/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_1/MatMulMatMul"simple_rnn_cell/Relu:activations:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_1/MatMul_1MatMulstates_11simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_2/MatMulMatMul$simple_rnn_cell_1/Relu:activations:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_2/MatMul_1MatMulstates_21simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� k
simple_rnn_cell_2/ReluRelusimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� s
IdentityIdentity$simple_rnn_cell_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� s

Identity_1Identity"simple_rnn_cell/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� u

Identity_2Identity$simple_rnn_cell_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� u

Identity_3Identity$simple_rnn_cell_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:���������:��������� :��������� :��������� : : : : : : : : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�s
�
while_body_73364
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0: W
Iwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorX
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: U
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  ��>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
.while/stacked_rnn_cells/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
/while/stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd8while/stacked_rnn_cells/simple_rnn_cell/MatMul:product:0Fwhile/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+while/stacked_rnn_cells/simple_rnn_cell/addAddV28while/stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:0:while/stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
,while/stacked_rnn_cells/simple_rnn_cell/ReluRelu/while/stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_3Iwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_1/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_1/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_4Iwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_2/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_2/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_6Identity<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp?^while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0"�
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0"�
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : 2�
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
@__inference_dense_layer_call_and_return_conditional_losses_73510

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�:
�	
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_72850

inputs

states
states_1
states_2@
.simple_rnn_cell_matmul_readvariableop_resource: =
/simple_rnn_cell_biasadd_readvariableop_resource: B
0simple_rnn_cell_matmul_1_readvariableop_resource:  B
0simple_rnn_cell_1_matmul_readvariableop_resource:  ?
1simple_rnn_cell_1_biasadd_readvariableop_resource: D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:  B
0simple_rnn_cell_2_matmul_readvariableop_resource:  ?
1simple_rnn_cell_2_biasadd_readvariableop_resource: D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity

identity_1

identity_2

identity_3��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�(simple_rnn_cell_1/BiasAdd/ReadVariableOp�'simple_rnn_cell_1/MatMul/ReadVariableOp�)simple_rnn_cell_1/MatMul_1/ReadVariableOp�(simple_rnn_cell_2/BiasAdd/ReadVariableOp�'simple_rnn_cell_2/MatMul/ReadVariableOp�)simple_rnn_cell_2/MatMul_1/ReadVariableOp�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
simple_rnn_cell/MatMulMatMulinputs-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell/MatMul_1MatMulstates/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_1/MatMulMatMul"simple_rnn_cell/Relu:activations:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_1/MatMul_1MatMulstates_11simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_2/MatMulMatMul$simple_rnn_cell_1/Relu:activations:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_2/MatMul_1MatMulstates_21simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� k
simple_rnn_cell_2/ReluRelusimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� s
IdentityIdentity$simple_rnn_cell_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� s

Identity_1Identity"simple_rnn_cell/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� u

Identity_2Identity$simple_rnn_cell_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� u

Identity_3Identity$simple_rnn_cell_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:���������:��������� :��������� :��������� : : : : : : : : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
'__inference_dense_1_layer_call_fn_74814

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_73526o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:%!

_user_specified_name74808:%!

_user_specified_name74810
�
�
%__inference_dense_layer_call_fn_74794

inputs
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_73510o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:%!

_user_specified_name74788:%!

_user_specified_name74790
��
�
__inference__traced_save_75119
file_prefix5
#read_disablecopyonread_dense_kernel: @1
#read_1_disablecopyonread_dense_bias:@9
'read_2_disablecopyonread_dense_1_kernel:@ 3
%read_3_disablecopyonread_dense_1_bias: 9
'read_4_disablecopyonread_dense_2_kernel: 3
%read_5_disablecopyonread_dense_2_bias:W
Eread_6_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_kernel: a
Oread_7_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_recurrent_kernel:  Q
Cread_8_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_bias: Y
Gread_9_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_1_kernel:  d
Rread_10_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_1_recurrent_kernel:  T
Fread_11_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_1_bias: Z
Hread_12_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_2_kernel:  d
Rread_13_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_2_recurrent_kernel:  T
Fread_14_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_2_bias: )
read_15_disablecopyonread_total: )
read_16_disablecopyonread_count: 
savev2_const
identity_35��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: @*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: @a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

: @w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@ y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

: y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnReadEread_6_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpEread_6_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_7/DisableCopyOnReadDisableCopyOnReadOread_7_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpOread_7_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_recurrent_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_8/DisableCopyOnReadDisableCopyOnReadCread_8_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpCread_8_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnReadGread_9_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpGread_9_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_1_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_10/DisableCopyOnReadDisableCopyOnReadRread_10_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_1_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpRread_10_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_1_recurrent_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_11/DisableCopyOnReadDisableCopyOnReadFread_11_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_1_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpFread_11_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_1_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnReadHread_12_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpHread_12_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_2_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_13/DisableCopyOnReadDisableCopyOnReadRread_13_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_2_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpRread_13_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_2_recurrent_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_14/DisableCopyOnReadDisableCopyOnReadFread_14_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_2_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpFread_14_disablecopyonread_rnn_stacked_rnn_cells_simple_rnn_cell_2_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_15/DisableCopyOnReadDisableCopyOnReadread_15_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpread_15_disablecopyonread_total^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_16/DisableCopyOnReadDisableCopyOnReadread_16_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpread_16_disablecopyonread_count^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 * 
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_34Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_35IdentityIdentity_34:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_35Identity_35:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_user_specified_namedense/kernel:*&
$
_user_specified_name
dense/bias:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_2/bias:LH
F
_user_specified_name.,rnn/stacked_rnn_cells/simple_rnn_cell/kernel:VR
P
_user_specified_name86rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel:J	F
D
_user_specified_name,*rnn/stacked_rnn_cells/simple_rnn_cell/bias:N
J
H
_user_specified_name0.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel:XT
R
_user_specified_name:8rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel:LH
F
_user_specified_name.,rnn/stacked_rnn_cells/simple_rnn_cell_1/bias:NJ
H
_user_specified_name0.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel:XT
R
_user_specified_name:8rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel:LH
F
_user_specified_name.,rnn/stacked_rnn_cells/simple_rnn_cell_2/bias:%!

_user_specified_nametotal:%!

_user_specified_namecount:=9

_output_shapes
: 

_user_specified_nameConst
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_73549
input_1
	rnn_73474: 
	rnn_73476: 
	rnn_73478:  
	rnn_73480:  
	rnn_73482: 
	rnn_73484:  
	rnn_73486:  
	rnn_73488: 
	rnn_73490:  
dense_73511: @
dense_73513:@
dense_1_73527:@ 
dense_1_73529: 
dense_2_73543: 
dense_2_73545:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallinput_1	rnn_73474	rnn_73476	rnn_73478	rnn_73480	rnn_73482	rnn_73484	rnn_73486	rnn_73488	rnn_73490*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_73473�
flatten/PartitionedCallPartitionedCall$rnn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_73498�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_73511dense_73513*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_73510�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73527dense_1_73529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_73526�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73543dense_2_73545*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_73542w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������: : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1:%!

_user_specified_name73474:%!

_user_specified_name73476:%!

_user_specified_name73478:%!

_user_specified_name73480:%!

_user_specified_name73482:%!

_user_specified_name73484:%!

_user_specified_name73486:%!

_user_specified_name73488:%	!

_user_specified_name73490:%
!

_user_specified_name73511:%!

_user_specified_name73513:%!

_user_specified_name73527:%!

_user_specified_name73529:%!

_user_specified_name73543:%!

_user_specified_name73545
�
�
while_cond_73080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_73080___redundant_placeholder03
/while_while_cond_73080___redundant_placeholder13
/while_while_cond_73080___redundant_placeholder23
/while_while_cond_73080___redundant_placeholder33
/while_while_cond_73080___redundant_placeholder43
/while_while_cond_73080___redundant_placeholder53
/while_while_cond_73080___redundant_placeholder63
/while_while_cond_73080___redundant_placeholder73
/while_while_cond_73080___redundant_placeholder83
/while_while_cond_73080___redundant_placeholder9
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k: : : : :��������� :��������� :��������� : :::::::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
�
while_cond_72877
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_72877___redundant_placeholder03
/while_while_cond_72877___redundant_placeholder13
/while_while_cond_72877___redundant_placeholder23
/while_while_cond_72877___redundant_placeholder33
/while_while_cond_72877___redundant_placeholder43
/while_while_cond_72877___redundant_placeholder53
/while_while_cond_72877___redundant_placeholder63
/while_while_cond_72877___redundant_placeholder73
/while_while_cond_72877___redundant_placeholder83
/while_while_cond_72877___redundant_placeholder9
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k: : : : :��������� :��������� :��������� : :::::::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
�
while_cond_74118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_74118___redundant_placeholder03
/while_while_cond_74118___redundant_placeholder13
/while_while_cond_74118___redundant_placeholder23
/while_while_cond_74118___redundant_placeholder33
/while_while_cond_74118___redundant_placeholder43
/while_while_cond_74118___redundant_placeholder53
/while_while_cond_74118___redundant_placeholder63
/while_while_cond_74118___redundant_placeholder73
/while_while_cond_74118___redundant_placeholder83
/while_while_cond_74118___redundant_placeholder9
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k: : : : :��������� :��������� :��������� : :::::::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�C
�
>__inference_rnn_layer_call_and_return_conditional_losses_73179

inputs)
stacked_rnn_cells_73054: %
stacked_rnn_cells_73056: )
stacked_rnn_cells_73058:  )
stacked_rnn_cells_73060:  %
stacked_rnn_cells_73062: )
stacked_rnn_cells_73064:  )
stacked_rnn_cells_73066:  %
stacked_rnn_cells_73068: )
stacked_rnn_cells_73070:  
identity��)stacked_rnn_cells/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
)stacked_rnn_cells/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0zeros_2:output:0stacked_rnn_cells_73054stacked_rnn_cells_73056stacked_rnn_cells_73058stacked_rnn_cells_73060stacked_rnn_cells_73062stacked_rnn_cells_73064stacked_rnn_cells_73066stacked_rnn_cells_73068stacked_rnn_cells_73070*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:��������� :��������� :��������� :��������� *+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_73053n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_73054stacked_rnn_cells_73056stacked_rnn_cells_73058stacked_rnn_cells_73060stacked_rnn_cells_73062stacked_rnn_cells_73064stacked_rnn_cells_73066stacked_rnn_cells_73068stacked_rnn_cells_73070*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*k
_output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *+
_read_only_resource_inputs
		
*
bodyR
while_body_73081*
condR
while_cond_73080*j
output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� V
NoOpNoOp*^stacked_rnn_cells/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:������������������: : : : : : : : : 2V
)stacked_rnn_cells/StatefulPartitionedCall)stacked_rnn_cells/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:%!

_user_specified_name73054:%!

_user_specified_name73056:%!

_user_specified_name73058:%!

_user_specified_name73060:%!

_user_specified_name73062:%!

_user_specified_name73064:%!

_user_specified_name73066:%!

_user_specified_name73068:%	!

_user_specified_name73070
�5
�
while_body_73081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_stacked_rnn_cells_73107_0: -
while_stacked_rnn_cells_73109_0: 1
while_stacked_rnn_cells_73111_0:  1
while_stacked_rnn_cells_73113_0:  -
while_stacked_rnn_cells_73115_0: 1
while_stacked_rnn_cells_73117_0:  1
while_stacked_rnn_cells_73119_0:  -
while_stacked_rnn_cells_73121_0: 1
while_stacked_rnn_cells_73123_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_stacked_rnn_cells_73107: +
while_stacked_rnn_cells_73109: /
while_stacked_rnn_cells_73111:  /
while_stacked_rnn_cells_73113:  +
while_stacked_rnn_cells_73115: /
while_stacked_rnn_cells_73117:  /
while_stacked_rnn_cells_73119:  +
while_stacked_rnn_cells_73121: /
while_stacked_rnn_cells_73123:  ��/while/stacked_rnn_cells/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
/while/stacked_rnn_cells/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_placeholder_4while_stacked_rnn_cells_73107_0while_stacked_rnn_cells_73109_0while_stacked_rnn_cells_73111_0while_stacked_rnn_cells_73113_0while_stacked_rnn_cells_73115_0while_stacked_rnn_cells_73117_0while_stacked_rnn_cells_73119_0while_stacked_rnn_cells_73121_0while_stacked_rnn_cells_73123_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:��������� :��������� :��������� :��������� *+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_73053r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/stacked_rnn_cells/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_6Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:3^while/NoOp*
T0*'
_output_shapes
:��������� Z

while/NoOpNoOp0^while/stacked_rnn_cells/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"@
while_stacked_rnn_cells_73107while_stacked_rnn_cells_73107_0"@
while_stacked_rnn_cells_73109while_stacked_rnn_cells_73109_0"@
while_stacked_rnn_cells_73111while_stacked_rnn_cells_73111_0"@
while_stacked_rnn_cells_73113while_stacked_rnn_cells_73113_0"@
while_stacked_rnn_cells_73115while_stacked_rnn_cells_73115_0"@
while_stacked_rnn_cells_73117while_stacked_rnn_cells_73117_0"@
while_stacked_rnn_cells_73119while_stacked_rnn_cells_73119_0"@
while_stacked_rnn_cells_73121while_stacked_rnn_cells_73121_0"@
while_stacked_rnn_cells_73123while_stacked_rnn_cells_73123_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : 2b
/while/stacked_rnn_cells/StatefulPartitionedCall/while/stacked_rnn_cells/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:%	!

_user_specified_name73107:%
!

_user_specified_name73109:%!

_user_specified_name73111:%!

_user_specified_name73113:%!

_user_specified_name73115:%!

_user_specified_name73117:%!

_user_specified_name73119:%!

_user_specified_name73121:%!

_user_specified_name73123
�
�
#__inference_rnn_layer_call_fn_74046

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5:  
	unknown_6: 
	unknown_7:  
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_73733o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:%!

_user_specified_name74026:%!

_user_specified_name74028:%!

_user_specified_name74030:%!

_user_specified_name74032:%!

_user_specified_name74034:%!

_user_specified_name74036:%!

_user_specified_name74038:%!

_user_specified_name74040:%	!

_user_specified_name74042
�
�
#__inference_signature_wrapper_73954
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: @
	unknown_9:@

unknown_10:@ 

unknown_11: 

unknown_12: 

unknown_13:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_72773o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1:%!

_user_specified_name73922:%!

_user_specified_name73924:%!

_user_specified_name73926:%!

_user_specified_name73928:%!

_user_specified_name73930:%!

_user_specified_name73932:%!

_user_specified_name73934:%!

_user_specified_name73936:%	!

_user_specified_name73938:%
!

_user_specified_name73940:%!

_user_specified_name73942:%!

_user_specified_name73944:%!

_user_specified_name73946:%!

_user_specified_name73948:%!

_user_specified_name73950
�s
�
while_body_74119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0: W
Iwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorX
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: U
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  ��>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
.while/stacked_rnn_cells/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
/while/stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd8while/stacked_rnn_cells/simple_rnn_cell/MatMul:product:0Fwhile/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+while/stacked_rnn_cells/simple_rnn_cell/addAddV28while/stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:0:while/stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
,while/stacked_rnn_cells/simple_rnn_cell/ReluRelu/while/stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_3Iwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_1/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_1/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_4Iwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_2/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_2/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_6Identity<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp?^while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0"�
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0"�
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : 2�
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�s
�
while_body_74483
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0: W
Iwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorX
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: U
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  ��>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
.while/stacked_rnn_cells/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
/while/stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd8while/stacked_rnn_cells/simple_rnn_cell/MatMul:product:0Fwhile/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+while/stacked_rnn_cells/simple_rnn_cell/addAddV28while/stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:0:while/stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
,while/stacked_rnn_cells/simple_rnn_cell/ReluRelu/while/stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_3Iwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_1/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_1/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_4Iwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_2/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_2/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_6Identity<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp?^while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0"�
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0"�
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : 2�
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
while_cond_73363
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_73363___redundant_placeholder03
/while_while_cond_73363___redundant_placeholder13
/while_while_cond_73363___redundant_placeholder23
/while_while_cond_73363___redundant_placeholder33
/while_while_cond_73363___redundant_placeholder43
/while_while_cond_73363___redundant_placeholder53
/while_while_cond_73363___redundant_placeholder63
/while_while_cond_73363___redundant_placeholder73
/while_while_cond_73363___redundant_placeholder83
/while_while_cond_73363___redundant_placeholder9
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k: : : : :��������� :��������� :��������� : :::::::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�x
�
>__inference_rnn_layer_call_and_return_conditional_losses_73473

inputsR
@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: O
Astacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: T
Bstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity��8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
(stacked_rnn_cells/simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd2stacked_rnn_cells/simple_rnn_cell/MatMul:product:0@stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulzeros:output:0Astacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%stacked_rnn_cells/simple_rnn_cell/addAddV22stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:04stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&stacked_rnn_cells/simple_rnn_cell/ReluRelu)stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul4stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulzeros_1:output:0Cstacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_1/addAddV24stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_1/ReluRelu+stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul6stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulzeros_2:output:0Cstacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_2/addAddV24stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_2/ReluRelu+stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*k
_output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *+
_read_only_resource_inputs
		
*
bodyR
while_body_73364*
condR
while_cond_73363*j
output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp9^stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8^stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2t
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource
�
�
*__inference_sequential_layer_call_fn_73840
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: @
	unknown_9:@

unknown_10:@ 

unknown_11: 

unknown_12: 

unknown_13:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_73770o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1:%!

_user_specified_name73808:%!

_user_specified_name73810:%!

_user_specified_name73812:%!

_user_specified_name73814:%!

_user_specified_name73816:%!

_user_specified_name73818:%!

_user_specified_name73820:%!

_user_specified_name73822:%	!

_user_specified_name73824:%
!

_user_specified_name73826:%!

_user_specified_name73828:%!

_user_specified_name73830:%!

_user_specified_name73832:%!

_user_specified_name73834:%!

_user_specified_name73836
�x
�
>__inference_rnn_layer_call_and_return_conditional_losses_73733

inputsR
@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: O
Astacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: T
Bstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity��8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
(stacked_rnn_cells/simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd2stacked_rnn_cells/simple_rnn_cell/MatMul:product:0@stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulzeros:output:0Astacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%stacked_rnn_cells/simple_rnn_cell/addAddV22stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:04stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&stacked_rnn_cells/simple_rnn_cell/ReluRelu)stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul4stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulzeros_1:output:0Cstacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_1/addAddV24stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_1/ReluRelu+stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul6stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulzeros_2:output:0Cstacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_2/addAddV24stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_2/ReluRelu+stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*k
_output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *+
_read_only_resource_inputs
		
*
bodyR
while_body_73624*
condR
while_cond_73623*j
output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp9^stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8^stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2t
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_73770
input_1
	rnn_73734: 
	rnn_73736: 
	rnn_73738:  
	rnn_73740:  
	rnn_73742: 
	rnn_73744:  
	rnn_73746:  
	rnn_73748: 
	rnn_73750:  
dense_73754: @
dense_73756:@
dense_1_73759:@ 
dense_1_73761: 
dense_2_73764: 
dense_2_73766:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallinput_1	rnn_73734	rnn_73736	rnn_73738	rnn_73740	rnn_73742	rnn_73744	rnn_73746	rnn_73748	rnn_73750*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_73733�
flatten/PartitionedCallPartitionedCall$rnn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_73498�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_73754dense_73756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_73510�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73759dense_1_73761*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_73526�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73764dense_2_73766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_73542w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������: : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1:%!

_user_specified_name73734:%!

_user_specified_name73736:%!

_user_specified_name73738:%!

_user_specified_name73740:%!

_user_specified_name73742:%!

_user_specified_name73744:%!

_user_specified_name73746:%!

_user_specified_name73748:%	!

_user_specified_name73750:%
!

_user_specified_name73754:%!

_user_specified_name73756:%!

_user_specified_name73759:%!

_user_specified_name73761:%!

_user_specified_name73764:%!

_user_specified_name73766
�:
�	
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_74995

inputs
states_0
states_1
states_2@
.simple_rnn_cell_matmul_readvariableop_resource: =
/simple_rnn_cell_biasadd_readvariableop_resource: B
0simple_rnn_cell_matmul_1_readvariableop_resource:  B
0simple_rnn_cell_1_matmul_readvariableop_resource:  ?
1simple_rnn_cell_1_biasadd_readvariableop_resource: D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:  B
0simple_rnn_cell_2_matmul_readvariableop_resource:  ?
1simple_rnn_cell_2_biasadd_readvariableop_resource: D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity

identity_1

identity_2

identity_3��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�(simple_rnn_cell_1/BiasAdd/ReadVariableOp�'simple_rnn_cell_1/MatMul/ReadVariableOp�)simple_rnn_cell_1/MatMul_1/ReadVariableOp�(simple_rnn_cell_2/BiasAdd/ReadVariableOp�'simple_rnn_cell_2/MatMul/ReadVariableOp�)simple_rnn_cell_2/MatMul_1/ReadVariableOp�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
simple_rnn_cell/MatMulMatMulinputs-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell/MatMul_1MatMulstates_0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_1/MatMulMatMul"simple_rnn_cell/Relu:activations:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_1/MatMul_1MatMulstates_11simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_2/MatMulMatMul$simple_rnn_cell_1/Relu:activations:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_2/MatMul_1MatMulstates_21simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� k
simple_rnn_cell_2/ReluRelusimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� s
IdentityIdentity$simple_rnn_cell_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� s

Identity_1Identity"simple_rnn_cell/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� u

Identity_2Identity$simple_rnn_cell_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� u

Identity_3Identity$simple_rnn_cell_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:���������:��������� :��������� :��������� : : : : : : : : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_1:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_2:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_73526

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_73498

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:��������� X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
B__inference_dense_2_layer_call_and_return_conditional_losses_74845

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�x
�
>__inference_rnn_layer_call_and_return_conditional_losses_74774

inputsR
@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: O
Astacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: T
Bstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity��8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
(stacked_rnn_cells/simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd2stacked_rnn_cells/simple_rnn_cell/MatMul:product:0@stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulzeros:output:0Astacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%stacked_rnn_cells/simple_rnn_cell/addAddV22stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:04stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&stacked_rnn_cells/simple_rnn_cell/ReluRelu)stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul4stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulzeros_1:output:0Cstacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_1/addAddV24stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_1/ReluRelu+stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul6stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulzeros_2:output:0Cstacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_2/addAddV24stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_2/ReluRelu+stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*k
_output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *+
_read_only_resource_inputs
		
*
bodyR
while_body_74665*
condR
while_cond_74664*j
output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp9^stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8^stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2t
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource
�
�
#__inference_rnn_layer_call_fn_73977
inputs_0
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5:  
	unknown_6: 
	unknown_7:  
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_72976o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:������������������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:%!

_user_specified_name73957:%!

_user_specified_name73959:%!

_user_specified_name73961:%!

_user_specified_name73963:%!

_user_specified_name73965:%!

_user_specified_name73967:%!

_user_specified_name73969:%!

_user_specified_name73971:%	!

_user_specified_name73973
�
�
while_cond_74482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_74482___redundant_placeholder03
/while_while_cond_74482___redundant_placeholder13
/while_while_cond_74482___redundant_placeholder23
/while_while_cond_74482___redundant_placeholder33
/while_while_cond_74482___redundant_placeholder43
/while_while_cond_74482___redundant_placeholder53
/while_while_cond_74482___redundant_placeholder63
/while_while_cond_74482___redundant_placeholder73
/while_while_cond_74482___redundant_placeholder83
/while_while_cond_74482___redundant_placeholder9
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k: : : : :��������� :��������� :��������� : :::::::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
��
�
sequential_rnn_while_body_72641:
6sequential_rnn_while_sequential_rnn_while_loop_counter@
<sequential_rnn_while_sequential_rnn_while_maximum_iterations$
 sequential_rnn_while_placeholder&
"sequential_rnn_while_placeholder_1&
"sequential_rnn_while_placeholder_2&
"sequential_rnn_while_placeholder_3&
"sequential_rnn_while_placeholder_49
5sequential_rnn_while_sequential_rnn_strided_slice_1_0u
qsequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensor_0i
Wsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0: f
Xsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0: k
Ysequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0:  k
Ysequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0:  h
Zsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0: m
[sequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:  k
Ysequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0:  h
Zsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0: m
[sequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:  !
sequential_rnn_while_identity#
sequential_rnn_while_identity_1#
sequential_rnn_while_identity_2#
sequential_rnn_while_identity_3#
sequential_rnn_while_identity_4#
sequential_rnn_while_identity_5#
sequential_rnn_while_identity_67
3sequential_rnn_while_sequential_rnn_strided_slice_1s
osequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensorg
Usequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: d
Vsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: i
Wsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  i
Wsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  f
Xsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: k
Ysequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  i
Wsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  f
Xsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: k
Ysequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  ��Msequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�Lsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�Nsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�Osequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�Nsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�Psequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�Osequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�Nsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�Psequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�
Fsequential/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
8sequential/rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqsequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensor_0 sequential_rnn_while_placeholderOsequential/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
Lsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpWsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
=sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMulMatMul?sequential/rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Tsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Msequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpXsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
>sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAddGsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul:product:0Usequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Nsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpYsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
?sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMul"sequential_rnn_while_placeholder_2Vsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/addAddV2Gsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:0Isequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
;sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/ReluRelu>sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
Nsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpYsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
?sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMulIsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Vsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Osequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpZsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
@sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAddIsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Wsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Psequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp[sequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
Asequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMul"sequential_rnn_while_placeholder_3Xsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
<sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/addAddV2Isequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:0Ksequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
=sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/ReluRelu@sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
Nsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpYsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
?sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMulKsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Vsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Osequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpZsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
@sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAddIsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Wsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Psequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp[sequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
Asequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMul"sequential_rnn_while_placeholder_4Xsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
<sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/addAddV2Isequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:0Ksequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
=sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/ReluRelu@sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� �
?sequential/rnn/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
9sequential/rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"sequential_rnn_while_placeholder_1Hsequential/rnn/while/TensorArrayV2Write/TensorListSetItem/index:output:0Ksequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0*
_output_shapes
: *
element_dtype0:���\
sequential/rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential/rnn/while/addAddV2 sequential_rnn_while_placeholder#sequential/rnn/while/add/y:output:0*
T0*
_output_shapes
: ^
sequential/rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential/rnn/while/add_1AddV26sequential_rnn_while_sequential_rnn_while_loop_counter%sequential/rnn/while/add_1/y:output:0*
T0*
_output_shapes
: �
sequential/rnn/while/IdentityIdentitysequential/rnn/while/add_1:z:0^sequential/rnn/while/NoOp*
T0*
_output_shapes
: �
sequential/rnn/while/Identity_1Identity<sequential_rnn_while_sequential_rnn_while_maximum_iterations^sequential/rnn/while/NoOp*
T0*
_output_shapes
: �
sequential/rnn/while/Identity_2Identitysequential/rnn/while/add:z:0^sequential/rnn/while/NoOp*
T0*
_output_shapes
: �
sequential/rnn/while/Identity_3IdentityIsequential/rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential/rnn/while/NoOp*
T0*
_output_shapes
: �
sequential/rnn/while/Identity_4IdentityIsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0^sequential/rnn/while/NoOp*
T0*'
_output_shapes
:��������� �
sequential/rnn/while/Identity_5IdentityKsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0^sequential/rnn/while/NoOp*
T0*'
_output_shapes
:��������� �
sequential/rnn/while/Identity_6IdentityKsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0^sequential/rnn/while/NoOp*
T0*'
_output_shapes
:��������� �
sequential/rnn/while/NoOpNoOpN^sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpM^sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpO^sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpP^sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpO^sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpQ^sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpP^sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpO^sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpQ^sequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 "G
sequential_rnn_while_identity&sequential/rnn/while/Identity:output:0"K
sequential_rnn_while_identity_1(sequential/rnn/while/Identity_1:output:0"K
sequential_rnn_while_identity_2(sequential/rnn/while/Identity_2:output:0"K
sequential_rnn_while_identity_3(sequential/rnn/while/Identity_3:output:0"K
sequential_rnn_while_identity_4(sequential/rnn/while/Identity_4:output:0"K
sequential_rnn_while_identity_5(sequential/rnn/while/Identity_5:output:0"K
sequential_rnn_while_identity_6(sequential/rnn/while/Identity_6:output:0"l
3sequential_rnn_while_sequential_rnn_strided_slice_15sequential_rnn_while_sequential_rnn_strided_slice_1_0"�
Xsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceZsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0"�
Ysequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource[sequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"�
Wsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceYsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0"�
Xsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceZsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0"�
Ysequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource[sequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"�
Wsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceYsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0"�
Vsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceXsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0"�
Wsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceYsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0"�
Usequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceWsequential_rnn_while_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0"�
osequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensorqsequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : 2�
Msequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpMsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2�
Lsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpLsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2�
Nsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpNsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2�
Osequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpOsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2�
Nsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpNsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2�
Psequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpPsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2�
Osequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpOsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2�
Nsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpNsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2�
Psequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpPsequential/rnn/while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:Y U

_output_shapes
: 
;
_user_specified_name#!sequential/rnn/while/loop_counter:_[

_output_shapes
: 
A
_user_specified_name)'sequential/rnn/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :VR

_output_shapes
: 
8
_user_specified_name sequential/rnn/strided_slice_1:nj

_output_shapes
: 
P
_user_specified_name86sequential/rnn/TensorArrayUnstack/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_sequential_layer_call_fn_73805
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: @
	unknown_9:@

unknown_10:@ 

unknown_11: 

unknown_12: 

unknown_13:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_73549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1:%!

_user_specified_name73773:%!

_user_specified_name73775:%!

_user_specified_name73777:%!

_user_specified_name73779:%!

_user_specified_name73781:%!

_user_specified_name73783:%!

_user_specified_name73785:%!

_user_specified_name73787:%	!

_user_specified_name73789:%
!

_user_specified_name73791:%!

_user_specified_name73793:%!

_user_specified_name73795:%!

_user_specified_name73797:%!

_user_specified_name73799:%!

_user_specified_name73801
�s
�
while_body_74301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0: W
Iwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorX
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: U
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  ��>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
.while/stacked_rnn_cells/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
/while/stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd8while/stacked_rnn_cells/simple_rnn_cell/MatMul:product:0Fwhile/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+while/stacked_rnn_cells/simple_rnn_cell/addAddV28while/stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:0:while/stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
,while/stacked_rnn_cells/simple_rnn_cell/ReluRelu/while/stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_3Iwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_1/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_1/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_4Iwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_2/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_2/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_6Identity<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp?^while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0"�
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0"�
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : 2�
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
#__inference_rnn_layer_call_fn_74000
inputs_0
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5:  
	unknown_6: 
	unknown_7:  
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_73179o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:������������������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:%!

_user_specified_name73980:%!

_user_specified_name73982:%!

_user_specified_name73984:%!

_user_specified_name73986:%!

_user_specified_name73988:%!

_user_specified_name73990:%!

_user_specified_name73992:%!

_user_specified_name73994:%	!

_user_specified_name73996
�x
�
>__inference_rnn_layer_call_and_return_conditional_losses_74592

inputsR
@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: O
Astacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: T
Bstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  T
Bstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  Q
Cstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: V
Dstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity��8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
(stacked_rnn_cells/simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd2stacked_rnn_cells/simple_rnn_cell/MatMul:product:0@stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulzeros:output:0Astacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%stacked_rnn_cells/simple_rnn_cell/addAddV22stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:04stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&stacked_rnn_cells/simple_rnn_cell/ReluRelu)stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul4stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulzeros_1:output:0Cstacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_1/addAddV24stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_1/ReluRelu+stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
*stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul6stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Astacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd4stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Bstacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
,stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulzeros_2:output:0Cstacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'stacked_rnn_cells/simple_rnn_cell_2/addAddV24stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:06stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
(stacked_rnn_cells/simple_rnn_cell_2/ReluRelu+stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceAstacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceBstacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceCstacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceDstacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*k
_output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *+
_read_only_resource_inputs
		
*
bodyR
while_body_74483*
condR
while_cond_74482*j
output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp9^stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8^stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;^stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:^stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp<^stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2t
8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp8stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp7stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2x
:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2v
9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp9stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2z
;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp;stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource
��
�
 __inference__wrapped_model_72773
input_1a
Osequential_rnn_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: ^
Psequential_rnn_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: c
Qsequential_rnn_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  c
Qsequential_rnn_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  `
Rsequential_rnn_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: e
Ssequential_rnn_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  c
Qsequential_rnn_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  `
Rsequential_rnn_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: e
Ssequential_rnn_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  A
/sequential_dense_matmul_readvariableop_resource: @>
0sequential_dense_biasadd_readvariableop_resource:@C
1sequential_dense_1_matmul_readvariableop_resource:@ @
2sequential_dense_1_biasadd_readvariableop_resource: C
1sequential_dense_2_matmul_readvariableop_resource: @
2sequential_dense_2_biasadd_readvariableop_resource:
identity��'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�)sequential/dense_2/BiasAdd/ReadVariableOp�(sequential/dense_2/MatMul/ReadVariableOp�Gsequential/rnn/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�Fsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�Hsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�Isequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�Hsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�Jsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�Isequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�Hsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�Jsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�sequential/rnn/whileY
sequential/rnn/ShapeShapeinput_1*
T0*
_output_shapes
::��l
"sequential/rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$sequential/rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$sequential/rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
sequential/rnn/strided_sliceStridedSlicesequential/rnn/Shape:output:0+sequential/rnn/strided_slice/stack:output:0-sequential/rnn/strided_slice/stack_1:output:0-sequential/rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
sequential/rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
sequential/rnn/zeros/packedPack%sequential/rnn/strided_slice:output:0&sequential/rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
sequential/rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential/rnn/zerosFill$sequential/rnn/zeros/packed:output:0#sequential/rnn/zeros/Const:output:0*
T0*'
_output_shapes
:��������� a
sequential/rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
sequential/rnn/zeros_1/packedPack%sequential/rnn/strided_slice:output:0(sequential/rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:a
sequential/rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential/rnn/zeros_1Fill&sequential/rnn/zeros_1/packed:output:0%sequential/rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� a
sequential/rnn/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
sequential/rnn/zeros_2/packedPack%sequential/rnn/strided_slice:output:0(sequential/rnn/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:a
sequential/rnn/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential/rnn/zeros_2Fill&sequential/rnn/zeros_2/packed:output:0%sequential/rnn/zeros_2/Const:output:0*
T0*'
_output_shapes
:��������� r
sequential/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential/rnn/transpose	Transposeinput_1&sequential/rnn/transpose/perm:output:0*
T0*+
_output_shapes
:���������p
sequential/rnn/Shape_1Shapesequential/rnn/transpose:y:0*
T0*
_output_shapes
::��n
$sequential/rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&sequential/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&sequential/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
sequential/rnn/strided_slice_1StridedSlicesequential/rnn/Shape_1:output:0-sequential/rnn/strided_slice_1/stack:output:0/sequential/rnn/strided_slice_1/stack_1:output:0/sequential/rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*sequential/rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
sequential/rnn/TensorArrayV2TensorListReserve3sequential/rnn/TensorArrayV2/element_shape:output:0'sequential/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Dsequential/rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6sequential/rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/rnn/transpose:y:0Msequential/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���n
$sequential/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&sequential/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&sequential/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
sequential/rnn/strided_slice_2StridedSlicesequential/rnn/transpose:y:0-sequential/rnn/strided_slice_2/stack:output:0/sequential/rnn/strided_slice_2/stack_1:output:0/sequential/rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
Fsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpOsequential_rnn_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
7sequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMulMatMul'sequential/rnn/strided_slice_2:output:0Nsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Gsequential/rnn/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpPsequential_rnn_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
8sequential/rnn/stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAddAsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul:product:0Osequential/rnn/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Hsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpQsequential_rnn_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
9sequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulsequential/rnn/zeros:output:0Psequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
4sequential/rnn/stacked_rnn_cells/simple_rnn_cell/addAddV2Asequential/rnn/stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:0Csequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
5sequential/rnn/stacked_rnn_cells/simple_rnn_cell/ReluRelu8sequential/rnn/stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
Hsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpQsequential_rnn_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
9sequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMulCsequential/rnn/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Psequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Isequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpRsequential_rnn_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
:sequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAddCsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Qsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Jsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpSsequential_rnn_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
;sequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulsequential/rnn/zeros_1:output:0Rsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
6sequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/addAddV2Csequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:0Esequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
7sequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/ReluRelu:sequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
Hsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpQsequential_rnn_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
9sequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMulEsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Psequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Isequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpRsequential_rnn_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
:sequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAddCsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Qsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Jsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpSsequential_rnn_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
;sequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulsequential/rnn/zeros_2:output:0Rsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
6sequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/addAddV2Csequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:0Esequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
7sequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/ReluRelu:sequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� }
,sequential/rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    m
+sequential/rnn/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
sequential/rnn/TensorArrayV2_1TensorListReserve5sequential/rnn/TensorArrayV2_1/element_shape:output:04sequential/rnn/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���U
sequential/rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'sequential/rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������c
!sequential/rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential/rnn/whileWhile*sequential/rnn/while/loop_counter:output:00sequential/rnn/while/maximum_iterations:output:0sequential/rnn/time:output:0'sequential/rnn/TensorArrayV2_1:handle:0sequential/rnn/zeros:output:0sequential/rnn/zeros_1:output:0sequential/rnn/zeros_2:output:0'sequential/rnn/strided_slice_1:output:0Fsequential/rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Osequential_rnn_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourcePsequential_rnn_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceQsequential_rnn_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceQsequential_rnn_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceRsequential_rnn_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceSsequential_rnn_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceQsequential_rnn_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceRsequential_rnn_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceSsequential_rnn_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*k
_output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *+
_read_only_resource_inputs
		
*+
body#R!
sequential_rnn_while_body_72641*+
cond#R!
sequential_rnn_while_cond_72640*j
output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *
parallel_iterations �
?sequential/rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
1sequential/rnn/TensorArrayV2Stack/TensorListStackTensorListStacksequential/rnn/while:output:3Hsequential/rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsw
$sequential/rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������p
&sequential/rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&sequential/rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
sequential/rnn/strided_slice_3StridedSlice:sequential/rnn/TensorArrayV2Stack/TensorListStack:tensor:0-sequential/rnn/strided_slice_3/stack:output:0/sequential/rnn/strided_slice_3/stack_1:output:0/sequential/rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskt
sequential/rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential/rnn/transpose_1	Transpose:sequential/rnn/TensorArrayV2Stack/TensorListStack:tensor:0(sequential/rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    �
sequential/flatten/ReshapeReshape'sequential/rnn/strided_slice_3:output:0!sequential/flatten/Const:output:0*
T0*'
_output_shapes
:��������� �
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� v
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
sequential/dense_2/TanhTanh#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentitysequential/dense_2/Tanh:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOpH^sequential/rnn/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpG^sequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpI^sequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpJ^sequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpI^sequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpK^sequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpJ^sequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpI^sequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpK^sequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp^sequential/rnn/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������: : : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2�
Gsequential/rnn/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpGsequential/rnn/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2�
Fsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpFsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2�
Hsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpHsequential/rnn/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2�
Isequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpIsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2�
Hsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpHsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2�
Jsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpJsequential/rnn/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2�
Isequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpIsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2�
Hsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpHsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2�
Jsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpJsequential/rnn/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp2,
sequential/rnn/whilesequential/rnn/while:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
1__inference_stacked_rnn_cells_layer_call_fn_74909

inputs
states_0
states_1
states_2
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5:  
	unknown_6: 
	unknown_7:  
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1states_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:��������� :��������� :��������� :��������� *+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_73053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:��������� q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:���������:��������� :��������� :��������� : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_1:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_2:%!

_user_specified_name74883:%!

_user_specified_name74885:%!

_user_specified_name74887:%!

_user_specified_name74889:%!

_user_specified_name74891:%	!

_user_specified_name74893:%
!

_user_specified_name74895:%!

_user_specified_name74897:%!

_user_specified_name74899
�

�
B__inference_dense_2_layer_call_and_return_conditional_losses_73542

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
C
'__inference_flatten_layer_call_fn_74779

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_73498`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�C
�
>__inference_rnn_layer_call_and_return_conditional_losses_72976

inputs)
stacked_rnn_cells_72851: %
stacked_rnn_cells_72853: )
stacked_rnn_cells_72855:  )
stacked_rnn_cells_72857:  %
stacked_rnn_cells_72859: )
stacked_rnn_cells_72861:  )
stacked_rnn_cells_72863:  %
stacked_rnn_cells_72865: )
stacked_rnn_cells_72867:  
identity��)stacked_rnn_cells/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
)stacked_rnn_cells/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0zeros_2:output:0stacked_rnn_cells_72851stacked_rnn_cells_72853stacked_rnn_cells_72855stacked_rnn_cells_72857stacked_rnn_cells_72859stacked_rnn_cells_72861stacked_rnn_cells_72863stacked_rnn_cells_72865stacked_rnn_cells_72867*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:��������� :��������� :��������� :��������� *+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_72850n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_72851stacked_rnn_cells_72853stacked_rnn_cells_72855stacked_rnn_cells_72857stacked_rnn_cells_72859stacked_rnn_cells_72861stacked_rnn_cells_72863stacked_rnn_cells_72865stacked_rnn_cells_72867*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*k
_output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *+
_read_only_resource_inputs
		
*
bodyR
while_body_72878*
condR
while_cond_72877*j
output_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� V
NoOpNoOp*^stacked_rnn_cells/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:������������������: : : : : : : : : 2V
)stacked_rnn_cells/StatefulPartitionedCall)stacked_rnn_cells/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:%!

_user_specified_name72851:%!

_user_specified_name72853:%!

_user_specified_name72855:%!

_user_specified_name72857:%!

_user_specified_name72859:%!

_user_specified_name72861:%!

_user_specified_name72863:%!

_user_specified_name72865:%	!

_user_specified_name72867
�s
�
while_body_73624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0: W
Iwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorX
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: U
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  ��>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
.while/stacked_rnn_cells/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
/while/stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd8while/stacked_rnn_cells/simple_rnn_cell/MatMul:product:0Fwhile/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+while/stacked_rnn_cells/simple_rnn_cell/addAddV28while/stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:0:while/stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
,while/stacked_rnn_cells/simple_rnn_cell/ReluRelu/while/stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_3Iwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_1/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_1/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_4Iwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_2/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_2/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_6Identity<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp?^while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0"�
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0"�
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : 2�
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�

sequential_rnn_while_cond_72640:
6sequential_rnn_while_sequential_rnn_while_loop_counter@
<sequential_rnn_while_sequential_rnn_while_maximum_iterations$
 sequential_rnn_while_placeholder&
"sequential_rnn_while_placeholder_1&
"sequential_rnn_while_placeholder_2&
"sequential_rnn_while_placeholder_3&
"sequential_rnn_while_placeholder_4<
8sequential_rnn_while_less_sequential_rnn_strided_slice_1Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder0Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder1Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder2Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder3Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder4Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder5Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder6Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder7Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder8Q
Msequential_rnn_while_sequential_rnn_while_cond_72640___redundant_placeholder9!
sequential_rnn_while_identity
�
sequential/rnn/while/LessLess sequential_rnn_while_placeholder8sequential_rnn_while_less_sequential_rnn_strided_slice_1*
T0*
_output_shapes
: i
sequential/rnn/while/IdentityIdentitysequential/rnn/while/Less:z:0*
T0
*
_output_shapes
: "G
sequential_rnn_while_identity&sequential/rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k: : : : :��������� :��������� :��������� : :::::::::::Y U

_output_shapes
: 
;
_user_specified_name#!sequential/rnn/while/loop_counter:_[

_output_shapes
: 
A
_user_specified_name)'sequential/rnn/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :VR

_output_shapes
: 
8
_user_specified_name sequential/rnn/strided_slice_1:

_output_shapes
:
�
�
while_cond_74664
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_74664___redundant_placeholder03
/while_while_cond_74664___redundant_placeholder13
/while_while_cond_74664___redundant_placeholder23
/while_while_cond_74664___redundant_placeholder33
/while_while_cond_74664___redundant_placeholder43
/while_while_cond_74664___redundant_placeholder53
/while_while_cond_74664___redundant_placeholder63
/while_while_cond_74664___redundant_placeholder73
/while_while_cond_74664___redundant_placeholder83
/while_while_cond_74664___redundant_placeholder9
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k: : : : :��������� :��������� :��������� : :::::::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
�
#__inference_rnn_layer_call_fn_74023

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2:  
	unknown_3: 
	unknown_4:  
	unknown_5:  
	unknown_6: 
	unknown_7:  
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_73473o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:%!

_user_specified_name74003:%!

_user_specified_name74005:%!

_user_specified_name74007:%!

_user_specified_name74009:%!

_user_specified_name74011:%!

_user_specified_name74013:%!

_user_specified_name74015:%!

_user_specified_name74017:%	!

_user_specified_name74019
�s
�
while_body_74665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0: W
Iwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:  \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0:  Y
Kwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0: ^
Lwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorX
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource: U
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource: Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource:  Z
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource:  W
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource: \
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource:  ��>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp�=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp�@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp�?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp�Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
.while/stacked_rnn_cells/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
/while/stacked_rnn_cells/simple_rnn_cell/BiasAddBiasAdd8while/stacked_rnn_cells/simple_rnn_cell/MatMul:product:0Fwhile/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+while/stacked_rnn_cells/simple_rnn_cell/addAddV28while/stacked_rnn_cells/simple_rnn_cell/BiasAdd:output:0:while/stacked_rnn_cells/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
,while/stacked_rnn_cells/simple_rnn_cell/ReluRelu/while/stacked_rnn_cells/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_1/MatMulMatMul:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_1/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_1/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_3Iwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_1/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_1/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
0while/stacked_rnn_cells/simple_rnn_cell_2/MatMulMatMul<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0Gwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
1while/stacked_rnn_cells/simple_rnn_cell_2/BiasAddBiasAdd:while/stacked_rnn_cells/simple_rnn_cell_2/MatMul:product:0Hwhile/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0�
2while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_4Iwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-while/stacked_rnn_cells/simple_rnn_cell_2/addAddV2:while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
.while/stacked_rnn_cells/simple_rnn_cell_2/ReluRelu1while/stacked_rnn_cells/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity:while/stacked_rnn_cells/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity<while/stacked_rnn_cells/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_6Identity<while/stacked_rnn_cells/simple_rnn_cell_2/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp?^while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpA^while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOpB^while/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_1_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_1_matmul_readvariableop_resource_0"�
Iwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resourceKwhile_stacked_rnn_cells_simple_rnn_cell_2_biasadd_readvariableop_resource_0"�
Jwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resourceLwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_2_matmul_readvariableop_resource_0"�
Gwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_simple_rnn_cell_biasadd_readvariableop_resource_0"�
Hwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_simple_rnn_cell_matmul_1_readvariableop_resource_0"�
Fwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : 2�
>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp>while/stacked_rnn_cells/simple_rnn_cell/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp=while/stacked_rnn_cells/simple_rnn_cell/MatMul/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_1/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_1/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_1/MatMul_1/ReadVariableOp2�
@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp@while/stacked_rnn_cells/simple_rnn_cell_2/BiasAdd/ReadVariableOp2�
?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp?while/stacked_rnn_cells/simple_rnn_cell_2/MatMul/ReadVariableOp2�
Awhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOpAwhile/stacked_rnn_cells/simple_rnn_cell_2/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
while_cond_74300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_74300___redundant_placeholder03
/while_while_cond_74300___redundant_placeholder13
/while_while_cond_74300___redundant_placeholder23
/while_while_cond_74300___redundant_placeholder33
/while_while_cond_74300___redundant_placeholder43
/while_while_cond_74300___redundant_placeholder53
/while_while_cond_74300___redundant_placeholder63
/while_while_cond_74300___redundant_placeholder73
/while_while_cond_74300___redundant_placeholder83
/while_while_cond_74300___redundant_placeholder9
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k: : : : :��������� :��������� :��������� : :::::::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�:
�	
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_74952

inputs
states_0
states_1
states_2@
.simple_rnn_cell_matmul_readvariableop_resource: =
/simple_rnn_cell_biasadd_readvariableop_resource: B
0simple_rnn_cell_matmul_1_readvariableop_resource:  B
0simple_rnn_cell_1_matmul_readvariableop_resource:  ?
1simple_rnn_cell_1_biasadd_readvariableop_resource: D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:  B
0simple_rnn_cell_2_matmul_readvariableop_resource:  ?
1simple_rnn_cell_2_biasadd_readvariableop_resource: D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:  
identity

identity_1

identity_2

identity_3��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�(simple_rnn_cell_1/BiasAdd/ReadVariableOp�'simple_rnn_cell_1/MatMul/ReadVariableOp�)simple_rnn_cell_1/MatMul_1/ReadVariableOp�(simple_rnn_cell_2/BiasAdd/ReadVariableOp�'simple_rnn_cell_2/MatMul/ReadVariableOp�)simple_rnn_cell_2/MatMul_1/ReadVariableOp�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
simple_rnn_cell/MatMulMatMulinputs-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell/MatMul_1MatMulstates_0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_1/MatMulMatMul"simple_rnn_cell/Relu:activations:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_1/MatMul_1MatMulstates_11simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:��������� k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:��������� �
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_2/MatMulMatMul$simple_rnn_cell_1/Relu:activations:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0�
simple_rnn_cell_2/MatMul_1MatMulstates_21simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:��������� k
simple_rnn_cell_2/ReluRelusimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:��������� s
IdentityIdentity$simple_rnn_cell_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� s

Identity_1Identity"simple_rnn_cell/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� u

Identity_2Identity$simple_rnn_cell_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� u

Identity_3Identity$simple_rnn_cell_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:���������:��������� :��������� :��������� : : : : : : : : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_1:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states_2:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
@__inference_dense_layer_call_and_return_conditional_losses_74805

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_74825

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�V
�
!__inference__traced_restore_75179
file_prefix/
assignvariableop_dense_kernel: @+
assignvariableop_1_dense_bias:@3
!assignvariableop_2_dense_1_kernel:@ -
assignvariableop_3_dense_1_bias: 3
!assignvariableop_4_dense_2_kernel: -
assignvariableop_5_dense_2_bias:Q
?assignvariableop_6_rnn_stacked_rnn_cells_simple_rnn_cell_kernel: [
Iassignvariableop_7_rnn_stacked_rnn_cells_simple_rnn_cell_recurrent_kernel:  K
=assignvariableop_8_rnn_stacked_rnn_cells_simple_rnn_cell_bias: S
Aassignvariableop_9_rnn_stacked_rnn_cells_simple_rnn_cell_1_kernel:  ^
Lassignvariableop_10_rnn_stacked_rnn_cells_simple_rnn_cell_1_recurrent_kernel:  N
@assignvariableop_11_rnn_stacked_rnn_cells_simple_rnn_cell_1_bias: T
Bassignvariableop_12_rnn_stacked_rnn_cells_simple_rnn_cell_2_kernel:  ^
Lassignvariableop_13_rnn_stacked_rnn_cells_simple_rnn_cell_2_recurrent_kernel:  N
@assignvariableop_14_rnn_stacked_rnn_cells_simple_rnn_cell_2_bias: #
assignvariableop_15_total: #
assignvariableop_16_count: 
identity_18��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp?assignvariableop_6_rnn_stacked_rnn_cells_simple_rnn_cell_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpIassignvariableop_7_rnn_stacked_rnn_cells_simple_rnn_cell_recurrent_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp=assignvariableop_8_rnn_stacked_rnn_cells_simple_rnn_cell_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpAassignvariableop_9_rnn_stacked_rnn_cells_simple_rnn_cell_1_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpLassignvariableop_10_rnn_stacked_rnn_cells_simple_rnn_cell_1_recurrent_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp@assignvariableop_11_rnn_stacked_rnn_cells_simple_rnn_cell_1_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpBassignvariableop_12_rnn_stacked_rnn_cells_simple_rnn_cell_2_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpLassignvariableop_13_rnn_stacked_rnn_cells_simple_rnn_cell_2_recurrent_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp@assignvariableop_14_rnn_stacked_rnn_cells_simple_rnn_cell_2_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_18IdentityIdentity_17:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_18Identity_18:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_user_specified_namedense/kernel:*&
$
_user_specified_name
dense/bias:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_2/bias:LH
F
_user_specified_name.,rnn/stacked_rnn_cells/simple_rnn_cell/kernel:VR
P
_user_specified_name86rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel:J	F
D
_user_specified_name,*rnn/stacked_rnn_cells/simple_rnn_cell/bias:N
J
H
_user_specified_name0.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel:XT
R
_user_specified_name:8rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel:LH
F
_user_specified_name.,rnn/stacked_rnn_cells/simple_rnn_cell_1/bias:NJ
H
_user_specified_name0.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel:XT
R
_user_specified_name:8rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel:LH
F
_user_specified_name.,rnn/stacked_rnn_cells/simple_rnn_cell_2/bias:%!

_user_specified_nametotal:%!

_user_specified_namecount
�
�
'__inference_dense_2_layer_call_fn_74834

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_73542o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:%!

_user_specified_name74828:%!

_user_specified_name74830
�
�
while_cond_73623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_13
/while_while_cond_73623___redundant_placeholder03
/while_while_cond_73623___redundant_placeholder13
/while_while_cond_73623___redundant_placeholder23
/while_while_cond_73623___redundant_placeholder33
/while_while_cond_73623___redundant_placeholder43
/while_while_cond_73623___redundant_placeholder53
/while_while_cond_73623___redundant_placeholder63
/while_while_cond_73623___redundant_placeholder73
/while_while_cond_73623___redundant_placeholder83
/while_while_cond_73623___redundant_placeholder9
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k: : : : :��������� :��������� :��������� : :::::::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�5
�
while_body_72878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_stacked_rnn_cells_72904_0: -
while_stacked_rnn_cells_72906_0: 1
while_stacked_rnn_cells_72908_0:  1
while_stacked_rnn_cells_72910_0:  -
while_stacked_rnn_cells_72912_0: 1
while_stacked_rnn_cells_72914_0:  1
while_stacked_rnn_cells_72916_0:  -
while_stacked_rnn_cells_72918_0: 1
while_stacked_rnn_cells_72920_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_stacked_rnn_cells_72904: +
while_stacked_rnn_cells_72906: /
while_stacked_rnn_cells_72908:  /
while_stacked_rnn_cells_72910:  +
while_stacked_rnn_cells_72912: /
while_stacked_rnn_cells_72914:  /
while_stacked_rnn_cells_72916:  +
while_stacked_rnn_cells_72918: /
while_stacked_rnn_cells_72920:  ��/while/stacked_rnn_cells/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
/while/stacked_rnn_cells/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_placeholder_4while_stacked_rnn_cells_72904_0while_stacked_rnn_cells_72906_0while_stacked_rnn_cells_72908_0while_stacked_rnn_cells_72910_0while_stacked_rnn_cells_72912_0while_stacked_rnn_cells_72914_0while_stacked_rnn_cells_72916_0while_stacked_rnn_cells_72918_0while_stacked_rnn_cells_72920_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:��������� :��������� :��������� :��������� *+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_72850r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/stacked_rnn_cells/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_6Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:3^while/NoOp*
T0*'
_output_shapes
:��������� Z

while/NoOpNoOp0^while/stacked_rnn_cells/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"@
while_stacked_rnn_cells_72904while_stacked_rnn_cells_72904_0"@
while_stacked_rnn_cells_72906while_stacked_rnn_cells_72906_0"@
while_stacked_rnn_cells_72908while_stacked_rnn_cells_72908_0"@
while_stacked_rnn_cells_72910while_stacked_rnn_cells_72910_0"@
while_stacked_rnn_cells_72912while_stacked_rnn_cells_72912_0"@
while_stacked_rnn_cells_72914while_stacked_rnn_cells_72914_0"@
while_stacked_rnn_cells_72916while_stacked_rnn_cells_72916_0"@
while_stacked_rnn_cells_72918while_stacked_rnn_cells_72918_0"@
while_stacked_rnn_cells_72920while_stacked_rnn_cells_72920_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :��������� :��������� :��������� : : : : : : : : : : : 2b
/while/stacked_rnn_cells/StatefulPartitionedCall/while/stacked_rnn_cells/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:%	!

_user_specified_name72904:%
!

_user_specified_name72906:%!

_user_specified_name72908:%!

_user_specified_name72910:%!

_user_specified_name72912:%!

_user_specified_name72914:%!

_user_specified_name72916:%!

_user_specified_name72918:%!

_user_specified_name72920"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������;
dense_20
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias"
_tf_keras_layer
�
40
51
62
73
84
95
:6
;7
<8
"9
#10
*11
+12
213
314"
trackable_list_wrapper
�
40
51
62
73
84
95
:6
;7
<8
"9
#10
*11
+12
213
314"
trackable_list_wrapper
 "
trackable_list_wrapper
�
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Btrace_0
Ctrace_12�
*__inference_sequential_layer_call_fn_73805
*__inference_sequential_layer_call_fn_73840�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zBtrace_0zCtrace_1
�
Dtrace_0
Etrace_12�
E__inference_sequential_layer_call_and_return_conditional_losses_73549
E__inference_sequential_layer_call_and_return_conditional_losses_73770�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0zEtrace_1
�B�
 __inference__wrapped_model_72773input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Fserving_default"
signature_map
_
40
51
62
73
84
95
:6
;7
<8"
trackable_list_wrapper
_
40
51
62
73
84
95
:6
;7
<8"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Gstates
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Mtrace_0
Ntrace_1
Otrace_2
Ptrace_32�
#__inference_rnn_layer_call_fn_73977
#__inference_rnn_layer_call_fn_74000
#__inference_rnn_layer_call_fn_74023
#__inference_rnn_layer_call_fn_74046�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zMtrace_0zNtrace_1zOtrace_2zPtrace_3
�
Qtrace_0
Rtrace_1
Strace_2
Ttrace_32�
>__inference_rnn_layer_call_and_return_conditional_losses_74228
>__inference_rnn_layer_call_and_return_conditional_losses_74410
>__inference_rnn_layer_call_and_return_conditional_losses_74592
>__inference_rnn_layer_call_and_return_conditional_losses_74774�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQtrace_0zRtrace_1zStrace_2zTtrace_3
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
	[cells"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
atrace_02�
'__inference_flatten_layer_call_fn_74779�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
�
btrace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_74785�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
htrace_02�
%__inference_dense_layer_call_fn_74794�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
�
itrace_02�
@__inference_dense_layer_call_and_return_conditional_losses_74805�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
: @2dense/kernel
:@2
dense/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
otrace_02�
'__inference_dense_1_layer_call_fn_74814�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0
�
ptrace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_74825�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
 :@ 2dense_1/kernel
: 2dense_1/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
vtrace_02�
'__inference_dense_2_layer_call_fn_74834�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0
�
wtrace_02�
B__inference_dense_2_layer_call_and_return_conditional_losses_74845�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0
 : 2dense_2/kernel
:2dense_2/bias
>:< 2,rnn/stacked_rnn_cells/simple_rnn_cell/kernel
H:F  26rnn/stacked_rnn_cells/simple_rnn_cell/recurrent_kernel
8:6 2*rnn/stacked_rnn_cells/simple_rnn_cell/bias
@:>  2.rnn/stacked_rnn_cells/simple_rnn_cell_1/kernel
J:H  28rnn/stacked_rnn_cells/simple_rnn_cell_1/recurrent_kernel
::8 2,rnn/stacked_rnn_cells/simple_rnn_cell_1/bias
@:>  2.rnn/stacked_rnn_cells/simple_rnn_cell_2/kernel
J:H  28rnn/stacked_rnn_cells/simple_rnn_cell_2/recurrent_kernel
::8 2,rnn/stacked_rnn_cells/simple_rnn_cell_2/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
x0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_sequential_layer_call_fn_73805input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_73840input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_73549input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_73770input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_73954input_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_1
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_rnn_layer_call_fn_73977inputs_0"�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_rnn_layer_call_fn_74000inputs_0"�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_rnn_layer_call_fn_74023inputs"�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_rnn_layer_call_fn_74046inputs"�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_rnn_layer_call_and_return_conditional_losses_74228inputs_0"�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_rnn_layer_call_and_return_conditional_losses_74410inputs_0"�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_rnn_layer_call_and_return_conditional_losses_74592inputs"�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_rnn_layer_call_and_return_conditional_losses_74774inputs"�
���
FullArgSpecG
args?�<
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
_
40
51
62
73
84
95
:6
;7
<8"
trackable_list_wrapper
_
40
51
62
73
84
95
:6
;7
<8"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
~trace_0
trace_12�
1__inference_stacked_rnn_cells_layer_call_fn_74877
1__inference_stacked_rnn_cells_layer_call_fn_74909�
���
FullArgSpec8
args0�-
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z~trace_0ztrace_1
�
�trace_0
�trace_12�
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_74952
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_74995�
���
FullArgSpec8
args0�-
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_flatten_layer_call_fn_74779inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_74785inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense_layer_call_fn_74794inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense_layer_call_and_return_conditional_losses_74805inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_1_layer_call_fn_74814inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_74825inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_2_layer_call_fn_74834inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_2_layer_call_and_return_conditional_losses_74845inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
 "
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_stacked_rnn_cells_layer_call_fn_74877inputsstates_0states_1states_2"�
���
FullArgSpec8
args0�-
jinputs
jstates
j	constants

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_stacked_rnn_cells_layer_call_fn_74909inputsstates_0states_1states_2"�
���
FullArgSpec8
args0�-
jinputs
jstates
j	constants

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_74952inputsstates_0states_1states_2"�
���
FullArgSpec8
args0�-
jinputs
jstates
j	constants

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_74995inputsstates_0states_1states_2"�
���
FullArgSpec8
args0�-
jinputs
jstates
j	constants

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

4kernel
5recurrent_kernel
6bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

7kernel
8recurrent_kernel
9bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

:kernel
;recurrent_kernel
<bias"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
5
70
81
92"
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
 __inference__wrapped_model_72773z465798:<;"#*+234�1
*�'
%�"
input_1���������
� "1�.
,
dense_2!�
dense_2����������
B__inference_dense_1_layer_call_and_return_conditional_losses_74825c*+/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
'__inference_dense_1_layer_call_fn_74814X*+/�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
B__inference_dense_2_layer_call_and_return_conditional_losses_74845c23/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
'__inference_dense_2_layer_call_fn_74834X23/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
@__inference_dense_layer_call_and_return_conditional_losses_74805c"#/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������@
� �
%__inference_dense_layer_call_fn_74794X"#/�,
%�"
 �
inputs��������� 
� "!�
unknown���������@�
B__inference_flatten_layer_call_and_return_conditional_losses_74785_/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� 
'__inference_flatten_layer_call_fn_74779T/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
>__inference_rnn_layer_call_and_return_conditional_losses_74228�	465798:<;S�P
I�F
4�1
/�,
inputs_0������������������

 
p

 

 
� ",�)
"�
tensor_0��������� 
� �
>__inference_rnn_layer_call_and_return_conditional_losses_74410�	465798:<;S�P
I�F
4�1
/�,
inputs_0������������������

 
p 

 

 
� ",�)
"�
tensor_0��������� 
� �
>__inference_rnn_layer_call_and_return_conditional_losses_74592~	465798:<;C�@
9�6
$�!
inputs���������

 
p

 

 
� ",�)
"�
tensor_0��������� 
� �
>__inference_rnn_layer_call_and_return_conditional_losses_74774~	465798:<;C�@
9�6
$�!
inputs���������

 
p 

 

 
� ",�)
"�
tensor_0��������� 
� �
#__inference_rnn_layer_call_fn_73977�	465798:<;S�P
I�F
4�1
/�,
inputs_0������������������

 
p

 

 
� "!�
unknown��������� �
#__inference_rnn_layer_call_fn_74000�	465798:<;S�P
I�F
4�1
/�,
inputs_0������������������

 
p 

 

 
� "!�
unknown��������� �
#__inference_rnn_layer_call_fn_74023s	465798:<;C�@
9�6
$�!
inputs���������

 
p

 

 
� "!�
unknown��������� �
#__inference_rnn_layer_call_fn_74046s	465798:<;C�@
9�6
$�!
inputs���������

 
p 

 

 
� "!�
unknown��������� �
E__inference_sequential_layer_call_and_return_conditional_losses_73549}465798:<;"#*+23<�9
2�/
%�"
input_1���������
p

 
� ",�)
"�
tensor_0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_73770}465798:<;"#*+23<�9
2�/
%�"
input_1���������
p 

 
� ",�)
"�
tensor_0���������
� �
*__inference_sequential_layer_call_fn_73805r465798:<;"#*+23<�9
2�/
%�"
input_1���������
p

 
� "!�
unknown����������
*__inference_sequential_layer_call_fn_73840r465798:<;"#*+23<�9
2�/
%�"
input_1���������
p 

 
� "!�
unknown����������
#__inference_signature_wrapper_73954�465798:<;"#*+23?�<
� 
5�2
0
input_1%�"
input_1���������"1�.
,
dense_2!�
dense_2����������
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_74952�	465798:<;���
���
 �
inputs���������
o�l
"�
states_0��������� 
"�
states_1��������� 
"�
states_2��������� 

 
p
� "���
���
$�!

tensor_0_0��������� 
{�x
&�#
tensor_0_1_0��������� 
&�#
tensor_0_1_1��������� 
&�#
tensor_0_1_2��������� 
� �
L__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_74995�	465798:<;���
���
 �
inputs���������
o�l
"�
states_0��������� 
"�
states_1��������� 
"�
states_2��������� 

 
p 
� "���
���
$�!

tensor_0_0��������� 
{�x
&�#
tensor_0_1_0��������� 
&�#
tensor_0_1_1��������� 
&�#
tensor_0_1_2��������� 
� �
1__inference_stacked_rnn_cells_layer_call_fn_74877�	465798:<;���
���
 �
inputs���������
o�l
"�
states_0��������� 
"�
states_1��������� 
"�
states_2��������� 

 
p
� "���
"�
tensor_0��������� 
u�r
$�!

tensor_1_0��������� 
$�!

tensor_1_1��������� 
$�!

tensor_1_2��������� �
1__inference_stacked_rnn_cells_layer_call_fn_74909�	465798:<;���
���
 �
inputs���������
o�l
"�
states_0��������� 
"�
states_1��������� 
"�
states_2��������� 

 
p 
� "���
"�
tensor_0��������� 
u�r
$�!

tensor_1_0��������� 
$�!

tensor_1_1��������� 
$�!

tensor_1_2��������� 