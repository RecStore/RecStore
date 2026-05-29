import tensorflow as tf
import os
import sys

# Add RecStore path
RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/framework/tensorflow/python_client'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

from client import RecstoreClient

class RecStoreEmbeddingLayer(tf.keras.layers.Layer):
    """
    TensorFlow Embedding layer using RecStore as backend
    Supports gradient updates during training
    """
    
    def __init__(self, 
                 embedding_dim,
                 name_prefix="recstore_emb",
                 library_path=None,
                 **kwargs):
        super(RecStoreEmbeddingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.name_prefix = name_prefix
        

        if library_path is None:
            library_path = '/home/wangyuexiang/RecStore/build/lib/lib_recstore_tf_ops.so'
        self.recstore_client = RecstoreClient(library_path)
        
    def build(self, input_shape):
        # No trainable parameters needed as embeddings are stored in RecStore
        super(RecStoreEmbeddingLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Forward pass: read embeddings from RecStore
        """
        # Check input shape
        input_shape = tf.shape(inputs)
        
        # If input is already embedding vectors (rank 2), use directly
        if len(inputs.shape) == 2 and inputs.shape[1] == self.embedding_dim:
            embeddings = inputs
        else:
            # If input is IDs (rank 1 or 2), read from RecStore
            if inputs.dtype != tf.uint64:
                inputs = tf.cast(inputs, tf.uint64)
            
            # Reshape input to rank 1 tensor
            inputs_flat = tf.reshape(inputs, [-1])
            
            # Read embeddings from RecStore
            embeddings = self.recstore_client.emb_read(inputs_flat)
            
            # Reshape embeddings back to correct shape
            batch_size = tf.shape(inputs)[0]
            embeddings = tf.reshape(embeddings, [batch_size, self.embedding_dim])
        
        # Register gradient function in training mode
        if training:
            embeddings = self._recstore_embedding_with_grad(inputs, embeddings)
            embeddings.set_shape([None, self.embedding_dim])
        
        return embeddings
    
    def _recstore_embedding_with_grad(self, inputs, embeddings):
        """
        RecStore embedding with custom gradient function
        Avoids tf.numpy_function, executes directly in TensorFlow graph
        """
        @tf.custom_gradient
        def _recstore_embedding_with_grad_inner(inputs_tensor, embeddings_tensor):
            def _grad_fn(grad_output):
                # Backward pass: update embeddings in RecStore
                batch_size = tf.shape(inputs_tensor)[0]
                
                # Reshape inputs to rank 1
                inputs_flat = tf.reshape(inputs_tensor, [-1])
                
                # Call RecStore emb_update
                update_op = self.recstore_client.emb_update(inputs_flat, grad_output)
                
                # Ensure update operation is executed
                with tf.control_dependencies([update_op]):
                    return tf.zeros_like(inputs_tensor), tf.zeros_like(embeddings_tensor)
            
            return embeddings_tensor, _grad_fn
        
        return _recstore_embedding_with_grad_inner(inputs, embeddings)

class RecStoreEmbeddingBag(tf.keras.layers.Layer):
    """
    RecStore EmbeddingBag with pooling operations
    """
    
    def __init__(self, 
                 embedding_dim,
                 pooling_mode="mean",
                 name_prefix="recstore_emb_bag",
                 library_path=None,
                 **kwargs):
        super(RecStoreEmbeddingBag, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.pooling_mode = pooling_mode
        self.name_prefix = name_prefix
        
        # Initialize RecStore client
        if library_path is None:
            library_path = '/home/wangyuexiang/RecStore/build/lib/lib_recstore_tf_ops.so'
        self.recstore_client = RecstoreClient(library_path)
        
    def build(self, input_shape):
        super(RecStoreEmbeddingBag, self).build(input_shape)
        
    def call(self, inputs, offsets=None, training=None):
        """
        Forward pass: read embeddings from RecStore and perform pooling
        """
        if offsets is None:
            return self._single_embedding_lookup(inputs, training)
        else:
            return self._pooled_embedding_lookup(inputs, offsets, training)
    
    def _single_embedding_lookup(self, inputs, training):
        """Single embedding lookup"""
        if inputs.dtype != tf.uint64:
            inputs = tf.cast(inputs, tf.uint64)
        
        embeddings = self.recstore_client.emb_read(inputs)
        
        if training:
            embeddings = tf.numpy_function(
                func=self._forward_with_gradient_single,
                inp=[inputs, embeddings],
                Tout=tf.float32,
                name=f"{self.name_prefix}_forward_single"
            )
            embeddings.set_shape([None, self.embedding_dim])
        
        return embeddings
    
    def _pooled_embedding_lookup(self, inputs, offsets, training):
        """Embedding lookup with pooling"""
        if inputs.dtype != tf.uint64:
            inputs = tf.cast(inputs, tf.uint64)
        
        # Read all embeddings from RecStore
        all_embeddings = self.recstore_client.emb_read(inputs)
        
        # Perform pooling based on offsets
        pooled_embeddings = self._pool_embeddings(all_embeddings, offsets)
        
        if training:
            pooled_embeddings = tf.numpy_function(
                func=self._forward_with_gradient_pooled,
                inp=[inputs, offsets, all_embeddings, pooled_embeddings],
                Tout=tf.float32,
                name=f"{self.name_prefix}_forward_pooled"
            )
            pooled_embeddings.set_shape([None, self.embedding_dim])
        
        return pooled_embeddings
    
    def _pool_embeddings(self, embeddings, offsets):
        """Perform embedding pooling based on pooling_mode"""
        if self.pooling_mode == "mean":
            return tf.math.unsorted_segment_mean(embeddings, offsets, tf.shape(offsets)[0] - 1)
        elif self.pooling_mode == "sum":
            return tf.math.unsorted_segment_sum(embeddings, offsets, tf.shape(offsets)[0] - 1)
        else:
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")
    
    def _forward_with_gradient_single(self, inputs, embeddings):
        """Forward pass and gradient registration for single embedding"""
        @tf.custom_gradient
        def _recstore_embedding_single_with_grad(inputs_tensor, embeddings_tensor):
            def _grad_fn(grad_output):
                self._update_embeddings(inputs_tensor, grad_output)
                return None, None
            return embeddings_tensor, _grad_fn
        
        return _recstore_embedding_single_with_grad(inputs, embeddings)
    
    def _forward_with_gradient_pooled(self, inputs, offsets, all_embeddings, pooled_embeddings):
        """Forward pass and gradient registration for pooled embedding"""
        @tf.custom_gradient
        def _recstore_embedding_pooled_with_grad(inputs_tensor, offsets_tensor, all_embeddings_tensor, pooled_embeddings_tensor):
            def _grad_fn(grad_output):
                # Distribute pooled gradients back to original embeddings
                expanded_grads = self._expand_pooled_gradients(grad_output, offsets_tensor, tf.shape(inputs_tensor)[0])
                self._update_embeddings(inputs_tensor, expanded_grads)
                return None, None, None, None
            return pooled_embeddings_tensor, _grad_fn
        
        return _recstore_embedding_pooled_with_grad(inputs, offsets, all_embeddings, pooled_embeddings)
    
    def _expand_pooled_gradients(self, pooled_grads, offsets, num_embeddings):
        """Expand pooled gradients to original embedding gradients"""
        # Simplified implementation: average gradient distribution
        expanded_grads = tf.tile(pooled_grads, [num_embeddings, 1])
        return expanded_grads / tf.cast(num_embeddings, tf.float32)
    
    def _update_embeddings(self, keys, gradients):
        """Update embeddings in RecStore"""
        try:
            update_op = self.recstore_client.emb_update(keys, gradients)
            return update_op
        except Exception as e:
            print(f"Warning: Failed to update embeddings: {e}")
            return None