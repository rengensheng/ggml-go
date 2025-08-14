package models

/*
#cgo CFLAGS: -I${SRCDIR}/../ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../ggml/build/src -lggml -lggml-base -lggml-cpu -lc++
#include <string.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
*/
import "C"

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"unsafe"

	"github.com/rengensheng/ggml-go/backend"
	"github.com/rengensheng/ggml-go/ctx"
	"github.com/rengensheng/ggml-go/lm"
	"github.com/rengensheng/ggml-go/tensor"
)

type HParams struct {
	NVocab        int32
	NMaxTokens    int32
	NEmbD         int32
	NIntermediate int32
	NHead         int32
	NLayer        int32
	F16           int32
}

type BertLayer struct {
	LnAttW *tensor.Tensor // ln_att_w
	LnAttB *tensor.Tensor // ln_att_b
	LnOutW *tensor.Tensor // ln_out_w
	LnOutB *tensor.Tensor // ln_out_b
	QW     *tensor.Tensor // q_w
	QB     *tensor.Tensor // q_b
	KW     *tensor.Tensor // k_w
	KB     *tensor.Tensor // k_b
	VW     *tensor.Tensor // v_w
	VB     *tensor.Tensor // v_b
	OW     *tensor.Tensor // o_w
	OB     *tensor.Tensor // o_b
	FFIW   *tensor.Tensor // ff_i_w
	FFIB   *tensor.Tensor // ff_i_b
	FFOW   *tensor.Tensor // ff_o_w
	FFOB   *tensor.Tensor // ff_o_b
}

type BertModel struct {
	WordEmbedding      *tensor.Tensor // word_embedding
	PositionEmbedding  *tensor.Tensor // position_embedding
	TokenTypeEmbedding *tensor.Tensor // token_type_embedding
	LayerNormWeight    *tensor.Tensor // layer_norm
	LayerNormBias      *tensor.Tensor // layer_norm_bias
	Layers             []*BertLayer
	ClsBias            *tensor.Tensor // cls_bias
	ClsDenseWeight     *tensor.Tensor // cls_dense
	ClsDenseBias       *tensor.Tensor // cls_dense_bias
	ClsLayerNormWeight *tensor.Tensor // cls_layer_norm
	ClsLayerNormBias   *tensor.Tensor // cls_layer_norm_bias
	ClsDecoderWeight   *tensor.Tensor // decode_weight
	ClsDecoderBias     *tensor.Tensor // decode_bias
}

func LoadBertModel(path string, context *ctx.Context) (*BertModel, *backend.Backend, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	var version uint32 = 1
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, nil, err
	}
	fmt.Println("version", version, version == 0x67676d6c)
	var hparams HParams
	if err := binary.Read(f, binary.LittleEndian, &hparams); err != nil {
		return nil, nil, err
	}
	fmt.Println("hparams", hparams)
	for i := 0; i < int(hparams.NVocab); i++ {
		var length int32
		err = binary.Read(f, binary.LittleEndian, &length)
		if err != nil {
			return nil, nil, err
		}
		word := make([]byte, length)
		err = binary.Read(f, binary.LittleEndian, &word)
		if err != nil {
			return nil, nil, err
		}
	}
	modelData := make(map[string]*tensor.Tensor)
	fmt.Println("n", hparams.NEmbD, hparams.NVocab)
	wType := lm.TypeQ4_0
	var model BertModel
	// a := tensor.NewTensor2D(context, lm.TypeQ4_0, int(hparams.NEmbD), int(hparams.NVocab))
	model.WordEmbedding = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), int(hparams.NVocab))
	modelData["bert.embeddings.word_embeddings.weight"] = model.WordEmbedding
	model.PositionEmbedding = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), int(hparams.NMaxTokens))
	modelData["bert.embeddings.position_embeddings.weight"] = model.PositionEmbedding
	model.TokenTypeEmbedding = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), 2)
	modelData["bert.embeddings.token_type_embeddings.weight"] = model.TokenTypeEmbedding
	model.LayerNormWeight = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
	modelData["bert.embeddings.LayerNorm.weight"] = model.LayerNormWeight
	model.LayerNormBias = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
	modelData["bert.embeddings.LayerNorm.bias"] = model.LayerNormBias
	for a := 0; a < int(hparams.NLayer); a++ {
		layer := &BertLayer{}
		// ---------- LayerNorm ----------
		layer.LnAttW = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
		layer.LnAttB = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
		layer.LnOutW = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
		layer.LnOutB = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
		// ---------- Q / K / V ----------
		// Q
		layer.QW = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), int(hparams.NEmbD))
		layer.QB = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
		// K
		layer.KW = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), int(hparams.NEmbD))
		layer.KB = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
		// V
		layer.VW = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), int(hparams.NEmbD))
		layer.VB = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))

		// ---------- O ----------
		layer.OW = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), int(hparams.NEmbD))
		layer.OB = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
		// ---------- Feed‑Forward ----------
		// ff_i
		layer.FFIW = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), int(hparams.NIntermediate))
		layer.FFIB = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NIntermediate))
		// ff_o
		layer.FFOW = tensor.NewTensor2D(context, wType, int(hparams.NIntermediate), int(hparams.NEmbD))
		layer.FFOB = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
		prefix := fmt.Sprintf("bert.encoder.layer.%d", a)
		modelData[prefix+".attention.self.query.weight"] = layer.QW
		modelData[prefix+".attention.self.query.bias"] = layer.QB
		modelData[prefix+".attention.self.key.weight"] = layer.KW
		modelData[prefix+".attention.self.key.bias"] = layer.KB
		modelData[prefix+".attention.self.value.weight"] = layer.VW
		modelData[prefix+".attention.self.value.bias"] = layer.VB
		modelData[prefix+".attention.output.dense.weight"] = layer.OW
		modelData[prefix+".attention.output.dense.bias"] = layer.OB
		modelData[prefix+".attention.output.LayerNorm.weight"] = layer.LnAttW
		modelData[prefix+".attention.output.LayerNorm.bias"] = layer.LnAttB
		modelData[prefix+".intermediate.dense.weight"] = layer.FFIW
		modelData[prefix+".intermediate.dense.bias"] = layer.FFIB
		modelData[prefix+".output.dense.weight"] = layer.FFOW
		modelData[prefix+".output.dense.bias"] = layer.FFOB
		modelData[prefix+".output.LayerNorm.weight"] = layer.LnOutW
		modelData[prefix+".output.LayerNorm.bias"] = layer.LnOutB
		model.Layers = append(model.Layers, layer)
	}
	// cls.predictions
	model.ClsBias = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NVocab))
	modelData["cls.predictions.bias"] = model.ClsBias
	model.ClsLayerNormWeight = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), int(hparams.NEmbD))
	modelData["cls.predictions.transform.dense.weight"] = model.ClsLayerNormWeight
	model.ClsLayerNormBias = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
	modelData["cls.predictions.transform.dense.bias"] = model.ClsLayerNormBias
	model.ClsLayerNormWeight = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
	modelData["cls.predictions.transform.LayerNorm.weight"] = model.ClsLayerNormWeight
	model.ClsLayerNormBias = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NEmbD))
	modelData["cls.predictions.transform.LayerNorm.bias"] = model.ClsLayerNormBias
	model.ClsDecoderWeight = tensor.NewTensor2D(context, wType, int(hparams.NEmbD), int(hparams.NVocab))
	modelData["cls.predictions.decoder.weight"] = model.ClsDecoderWeight
	model.ClsDecoderBias = tensor.NewTensor1D(context, lm.TypeF32, int(hparams.NVocab))
	modelData["cls.predictions.decoder.bias"] = model.ClsDecoderBias
	end := backend.NewCPUBackend()
	end.BackendAllocCtxTensors(context)
	for {
		var nDims int32
		var nLength int32
		var fType int32
		err := binary.Read(f, binary.LittleEndian, &nDims)
		if err != nil {
			if errors.Is(err, io.EOF) {
				log.Println("The file read to the end of the file.")
				return &model, end, nil
			}
			return nil, nil, err
		}
		err = binary.Read(f, binary.LittleEndian, &nLength)
		if err != nil {
			return nil, nil, err
		}
		err = binary.Read(f, binary.LittleEndian, &fType)
		if err != nil {
			return nil, nil, err
		}
		fmt.Println("size", nDims, nLength, fType)
		nElements := 1
		ne := make([]int, 10)
		for j := 0; j < int(nDims); j++ {
			var neCur int32
			err = binary.Read(f, binary.LittleEndian, &neCur)
			if err != nil {
				return nil, nil, err
			}
			fmt.Println("neCur", neCur)
			ne[j] = int(neCur)
			nElements *= int(neCur)
		}
		nameBytes := make([]byte, nLength)
		err = binary.Read(f, binary.LittleEndian, &nameBytes)
		if err != nil {
			return nil, nil, err
		}
		name := string(nameBytes)
		currentTensor := modelData[name]
		if nElements != int(currentTensor.NElements()) {
			return nil, nil, fmt.Errorf("tensor size mismatch: %d != %d", nElements, name, currentTensor.NElements())
		}
		output := make([]byte, currentTensor.NBytes())
		err = binary.Read(f, binary.LittleEndian, output)
		if err != nil {
			return nil, nil, err
		}
		length := currentTensor.NBytes()
		end.TensorSet(currentTensor, unsafe.Pointer(&output[0]), 0, length)
		// currentTensor.ReadTensorData(f)
	}
}
