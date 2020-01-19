// This file is MACHINE GENERATED! Do not edit.

#ifndef C__USERS_USER_SOURCE_REPOS_TENSORFLOW_TENSORFLOW_CONTRIB_CMAKE_BUILD_TENSORFLOW_CC_OPS_DATASET_OPS_INTERNAL_H_
#define C__USERS_USER_SOURCE_REPOS_TENSORFLOW_TENSORFLOW_CONTRIB_CMAKE_BUILD_TENSORFLOW_CC_OPS_DATASET_OPS_INTERNAL_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

/// @defgroup dataset_ops_internal Dataset Ops Internal
/// @{

/// Creates a dataset that batches `batch_size` elements from `input_dataset`.
///
/// Arguments:
/// * scope: A Scope object
/// * batch_size: A scalar representing the number of elements to accumulate in a batch.
/// * drop_remainder: A scalar representing whether the last batch should be dropped in case its size
/// is smaller than desired.
///
/// Returns:
/// * `Output`: The handle tensor.
class BatchDatasetV2 {
 public:
  BatchDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset, ::tensorflow::Input batch_size,
               ::tensorflow::Input drop_remainder, const DataTypeSlice&
               output_types, const gtl::ArraySlice<PartialTensorShape>&
               output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  ::tensorflow::Output handle;
};

/// Returns a serialized GraphDef representing `input_dataset`.
///
/// Returns a graph representation for `input_dataset`.
///
/// Arguments:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the dataset to return the graph representation for.
///
/// Returns:
/// * `Output`: The graph representation of the dataset (as serialized GraphDef).
class DatasetToGraph {
 public:
  DatasetToGraph(const ::tensorflow::Scope& scope, ::tensorflow::Input
               input_dataset);
  operator ::tensorflow::Output() const { return graph; }
  operator ::tensorflow::Input() const { return graph; }
  ::tensorflow::Node* node() const { return graph.node(); }

  ::tensorflow::Output graph;
};

/// Writes the given dataset to the given file using the TFRecord format.
///
/// Arguments:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the dataset to write.
/// * filename: A scalar string tensor representing the filename to use.
/// * compression_type: A scalar string tensor containing either (i) the empty string (no
/// compression), (ii) "ZLIB", or (iii) "GZIP".
///
/// Returns:
/// * the created `Operation`
class DatasetToTFRecord {
 public:
  DatasetToTFRecord(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  input_dataset, ::tensorflow::Input filename,
                  ::tensorflow::Input compression_type);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Creates a dataset that computes a group-by on `input_dataset`.
///
/// Creates a dataset that computes a group-by on `input_dataset`.
///
/// Arguments:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * key_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `key_func`.
/// * init_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `init_func`.
/// * reduce_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `reduce_func`.
/// * finalize_func_other_arguments: A list of tensors, typically values that were captured when
/// building a closure for `finalize_func`.
/// * key_func: A function mapping an element of `input_dataset`, concatenated
/// with `key_func_other_arguments` to a scalar value of type DT_INT64.
/// * init_func: A function mapping a key of type DT_INT64, concatenated with
/// `init_func_other_arguments` to the initial reducer state.
/// * reduce_func: A function mapping the current reducer state and an element of `input_dataset`,
/// concatenated with `reduce_func_other_arguments` to a new reducer state.
/// * finalize_func: A function mapping the final reducer state to an output element.
///
/// Returns:
/// * `Output`: The handle tensor.
class GroupByReducerDataset {
 public:
  GroupByReducerDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_dataset, ::tensorflow::InputList
                      key_func_other_arguments, ::tensorflow::InputList
                      init_func_other_arguments, ::tensorflow::InputList
                      reduce_func_other_arguments, ::tensorflow::InputList
                      finalize_func_other_arguments, const NameAttrList&
                      key_func, const NameAttrList& init_func, const
                      NameAttrList& reduce_func, const NameAttrList&
                      finalize_func, const DataTypeSlice& output_types, const
                      gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  ::tensorflow::Output handle;
};

/// TODO: add doc.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The resource_handle tensor.
class IteratorFromStringHandleV2 {
 public:
  /// Optional attribute setters for IteratorFromStringHandleV2
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.output_types_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.output_shapes_ = x;
      return ret;
    }

    DataTypeSlice output_types_ = {};
    gtl::ArraySlice<PartialTensorShape> output_shapes_ = {};
  };
  IteratorFromStringHandleV2(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input string_handle);
  IteratorFromStringHandleV2(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input string_handle, const
                           IteratorFromStringHandleV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return resource_handle; }
  operator ::tensorflow::Input() const { return resource_handle; }
  ::tensorflow::Node* node() const { return resource_handle.node(); }

  static Attrs OutputTypes(const DataTypeSlice& x) {
    return Attrs().OutputTypes(x);
  }
  static Attrs OutputShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().OutputShapes(x);
  }

  ::tensorflow::Output resource_handle;
};

/// TODO: add doc.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The handle tensor.
class IteratorV2 {
 public:
  IteratorV2(const ::tensorflow::Scope& scope, StringPiece shared_name,
           StringPiece container, const DataTypeSlice& output_types, const
           gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  ::tensorflow::Output handle;
};

/// Creates a dataset that fuses mapping with batching.
///
/// Creates a dataset that applies `f` to the outputs of `input_dataset` and then
/// batches `batch_size` of them.
/// 
/// Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
/// to `batch_size * num_parallel_batches` copies of `f` in parallel.
///
/// Arguments:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * other_arguments: A list of tensors, typically values that were captured when building a closure
/// for `f`.
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch. It determines the number of concurrent invocations of `f` that process
/// elements from `input_dataset` in parallel.
/// * num_parallel_batches: A scalar representing the number of batches to create in parallel. Processing
/// multiple batches in parallel benefits workloads prone to stragglers.
/// * drop_remainder: A scalar representing whether the last batch should be dropped in case its size
/// is smaller than desired.
/// * f: A function to apply to the outputs of `input_dataset`.
///
/// Returns:
/// * `Output`: The handle tensor.
class MapAndBatchDataset {
 public:
  MapAndBatchDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   input_dataset, ::tensorflow::InputList other_arguments,
                   ::tensorflow::Input batch_size, ::tensorflow::Input
                   num_parallel_batches, ::tensorflow::Input drop_remainder,
                   const NameAttrList& f, const DataTypeSlice& output_types,
                   const gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  ::tensorflow::Output handle;
};

/// Creates a dataset that fuses mapping with batching.
///
/// Creates a dataset that applies `f` to the outputs of `input_dataset` and then
/// batches `batch_size` of them.
/// 
/// Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
/// to `batch_size * num_parallel_batches` copies of `f` in parallel.
///
/// Arguments:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * other_arguments: A list of tensors, typically values that were captured when building a closure
/// for `f`.
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch. It determines the number of concurrent invocations of `f` that process
/// elements from `input_dataset` in parallel.
/// * num_parallel_calls: A scalar representing the maximum number of parallel invocations of the `map_fn`
/// function. Applying the `map_fn` on consecutive input elements in parallel has
/// the potential to improve input pipeline throughput.
/// * drop_remainder: A scalar representing whether the last batch should be dropped in case its size
/// is smaller than desired.
/// * f: A function to apply to the outputs of `input_dataset`.
///
/// Returns:
/// * `Output`: The handle tensor.
class MapAndBatchDatasetV2 {
 public:
  MapAndBatchDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::InputList other_arguments,
                     ::tensorflow::Input batch_size, ::tensorflow::Input
                     num_parallel_calls, ::tensorflow::Input drop_remainder,
                     const NameAttrList& f, const DataTypeSlice& output_types,
                     const gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  ::tensorflow::Output handle;
};

/// Creates a dataset by applying optimizations to `input_dataset`.
///
/// Creates a dataset by applying optimizations to `input_dataset`.
///
/// Arguments:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
/// * optimizations: A `tf.string` vector `tf.Tensor` identifying optimizations to use.
///
/// Returns:
/// * `Output`: The handle tensor.
class OptimizeDataset {
 public:
  OptimizeDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_dataset, ::tensorflow::Input optimizations, const
                DataTypeSlice& output_types, const
                gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  ::tensorflow::Output handle;
};

/// Creates a dataset that batches and pads `batch_size` elements from the input.
///
/// Arguments:
/// * scope: A Scope object
/// * batch_size: A scalar representing the number of elements to accumulate in a
/// batch.
/// * padded_shapes: A list of int64 tensors representing the desired padded shapes
/// of the corresponding output components. These shapes may be partially
/// specified, using `-1` to indicate that a particular dimension should be
/// padded to the maximum size of all batch elements.
/// * padding_values: A list of scalars containing the padding value to use for
/// each of the outputs.
/// * drop_remainder: A scalar representing whether the last batch should be dropped in case its size
/// is smaller than desired.
///
/// Returns:
/// * `Output`: The handle tensor.
class PaddedBatchDatasetV2 {
 public:
  PaddedBatchDatasetV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input_dataset, ::tensorflow::Input batch_size,
                     ::tensorflow::InputList padded_shapes,
                     ::tensorflow::InputList padding_values,
                     ::tensorflow::Input drop_remainder, const
                     gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  ::tensorflow::Output handle;
};

/// A placeholder for input pipeline graph optimizations.
///
/// A placeholder for input pipeline graph optimizations.
///
/// Arguments:
/// * scope: A Scope object
/// * input_dataset: A variant tensor representing the input dataset.
///
/// Returns:
/// * `Output`: The handle tensor.
class SinkDataset {
 public:
  SinkDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
            input_dataset);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  ::tensorflow::Output handle;
};

/// A dataset that creates window datasets from the input dataset.
///
/// Arguments:
/// * scope: A Scope object
/// * window_size: A scalar representing the number of elements to accumulate in a window.
///
/// Returns:
/// * `Output`: The handle tensor.
class WindowDataset {
 public:
  WindowDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_dataset, ::tensorflow::Input window_size, const
              DataTypeSlice& output_types, const
              gtl::ArraySlice<PartialTensorShape>& output_shapes);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  ::tensorflow::Output handle;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // C__USERS_USER_SOURCE_REPOS_TENSORFLOW_TENSORFLOW_CONTRIB_CMAKE_BUILD_TENSORFLOW_CC_OPS_DATASET_OPS_INTERNAL_H_
