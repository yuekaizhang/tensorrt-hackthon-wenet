import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def addCast(graph, first_input_tensor='552', second_input_tensor='613'):
  # first_input_tensor: '552', second_input_tensor: '613' for small conformer offline encoder
  # first_input_tensor: '516', second_input_tensor: '577' for large conformer offline encoder

  tmap = graph.tensors()

  firstSliceNode = [node for node in graph.nodes if node.name == "Slice_79"][0]
  tmap[first_input_tensor].outputs.clear()
  castOut = gs.Variable(name="castout")
  castor = gs.Node(op="Cast",inputs=[tmap[first_input_tensor]],outputs=[castOut])
  castor.attrs["to"] = onnx.TensorProto.INT64
  graph.nodes.append(castor)
  firstSliceNode.inputs.insert(0, castOut)

  secondSliceNode = [node for node in graph.nodes if node.name == "Slice_84"][0]
  tmap[second_input_tensor].inputs.clear()
  castOut2 = gs.Variable(name="castout2")
  castor2 = gs.Node(op="Cast",inputs=[castOut2],outputs=[tmap[second_input_tensor]])
  castor2.attrs["to"] = onnx.TensorProto.BOOL
  graph.nodes.append(castor2)
  secondSliceNode.outputs = [castOut2]

  # Remove the now-dangling subgraph.
  graph.cleanup().toposort()
  return graph

def replaceLayerNorm(graph):
  nodesLayerNorm=0
  for node in graph.nodes:
    if node.op == 'Div':
        if node.inputs[0].inputs[0].inputs[1].inputs[0].op == 'ReduceMean':
            nodesLayerNorm += 1
            pluginVariable = gs.Variable(f"CustomLayerNorm{nodesLayerNorm}", np.dtype(np.float32), None)
            pluginNode = gs.Node("LayerNorm", f"CustomLayerNorm{nodesLayerNorm}", inputs=[node.i(0).i(0).outputs[0]],
                                                                                  outputs=[pluginVariable], 
                                                                                  attrs={"bias": 1e-3})
            graph.nodes.append(pluginNode)
            node.o().inputs[0] = pluginVariable
            node.outputs.clear()
  graph.cleanup().toposort()
  return graph

def swpDiv(graph):
  # sol 1 https://github.com/nvidia-china-sae/CISI/issues/18
  i = 0
  for node in graph.nodes:
    if node.op == 'Div' and 'My' not in node.name:
        i += 1 
        nodeAdd = node.i()
        if  not len(nodeAdd.inputs) == 2:
            continue
        nodeAdd.outputs = node.outputs
        node.outputs.clear()

        DivOut = gs.Variable("div_out" + str(i))
        Div2Out = gs.Variable("div2_out" + str(i))
              
        numberDiv = gs.Constant("denominator" + str(i), np.array([8], dtype=np.float32))
        nodeDiv = gs.Node("Div", "MyDiv" + str(i), inputs=[nodeAdd.inputs[0], numberDiv], outputs=[DivOut])
        numberDiv2 = gs.Constant("denominator2" + str(i), np.array([8], dtype=np.float32))
        nodeDiv2 = gs.Node("Div", "MyDiv2" + str(i), inputs=[nodeAdd.inputs[1], numberDiv2], outputs=[Div2Out])

        nodeAdd.inputs = [DivOut, Div2Out]
        graph.nodes.append(nodeDiv)
        graph.nodes.append(nodeDiv2)
  graph.cleanup().toposort()
  return graph

@gs.Graph.register()
def replace_attn(self, inputs, outputs, name, attrs = None):
   
    for out in outputs:
        out.inputs.clear()


    if attrs:
        return self.layer(op="DecoderAttention", 
                    inputs=inputs, 
                    outputs=outputs,
                    name=name,
                    attrs=attrs
                    )
    return self.layer(op="Attention", 
                    inputs=inputs, 
                    outputs=outputs,
                    name=name,
                    )

enc_self_attn_nodes = [
        {"inps" : ['646',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3215',            #   pos_emb weight
                   '3203','encoder.encoders.0.self_attn.linear_q.bias',
                   '3207','encoder.encoders.0.self_attn.linear_k.bias',
                   '3211','encoder.encoders.0.self_attn.linear_v.bias',
                   '3221','encoder.encoders.0.self_attn.linear_out.bias',
                   'encoder.encoders.0.self_attn.pos_bias_u',
                   'encoder.encoders.0.self_attn.pos_bias_v',
                ],
        "outs" : ['757']},
        {"inps" : ['856',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3238',            #   pos_emb weight
                   '3226','encoder.encoders.1.self_attn.linear_q.bias',
                   '3230','encoder.encoders.1.self_attn.linear_k.bias',
                   '3234','encoder.encoders.1.self_attn.linear_v.bias',
                   '3244','encoder.encoders.1.self_attn.linear_out.bias',
                   'encoder.encoders.1.self_attn.pos_bias_u',
                   'encoder.encoders.1.self_attn.pos_bias_v',
                ],
        "outs" : ['967']},
        {"inps" : ['1066',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3261',            #   pos_emb weight
                   '3249','encoder.encoders.2.self_attn.linear_q.bias',
                   '3253','encoder.encoders.2.self_attn.linear_k.bias',
                   '3257','encoder.encoders.2.self_attn.linear_v.bias',
                   '3267','encoder.encoders.2.self_attn.linear_out.bias',
                   'encoder.encoders.2.self_attn.pos_bias_u',
                   'encoder.encoders.2.self_attn.pos_bias_v',
                ],
        "outs" : ['1177']},
        {"inps" : ['1276',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3284',            #   pos_emb weight
                   '3272','encoder.encoders.3.self_attn.linear_q.bias',
                   '3276','encoder.encoders.3.self_attn.linear_k.bias',
                   '3280','encoder.encoders.3.self_attn.linear_v.bias',
                   '3290','encoder.encoders.3.self_attn.linear_out.bias',
                   'encoder.encoders.3.self_attn.pos_bias_u',
                   'encoder.encoders.3.self_attn.pos_bias_v',
                ],
        "outs" : ['1387']},
        {"inps" : ['1486',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3307',            #   pos_emb weight
                   '3295','encoder.encoders.4.self_attn.linear_q.bias',
                   '3299','encoder.encoders.4.self_attn.linear_k.bias',
                   '3303','encoder.encoders.4.self_attn.linear_v.bias',
                   '3313','encoder.encoders.4.self_attn.linear_out.bias',
                   'encoder.encoders.4.self_attn.pos_bias_u',
                   'encoder.encoders.4.self_attn.pos_bias_v',
                ],
        "outs" : ['1597']},
        {"inps" : ['1696',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3330',            #   pos_emb weight
                   '3318','encoder.encoders.5.self_attn.linear_q.bias',
                   '3322','encoder.encoders.5.self_attn.linear_k.bias',
                   '3326','encoder.encoders.5.self_attn.linear_v.bias',
                   '3336','encoder.encoders.5.self_attn.linear_out.bias',
                   'encoder.encoders.5.self_attn.pos_bias_u',
                   'encoder.encoders.5.self_attn.pos_bias_v',
                ],
        "outs" : ['1807']},
        {"inps" : ['1906',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3353',            #   pos_emb weight
                   '3341','encoder.encoders.6.self_attn.linear_q.bias',
                   '3345','encoder.encoders.6.self_attn.linear_k.bias',
                   '3349','encoder.encoders.6.self_attn.linear_v.bias',
                   '3359','encoder.encoders.6.self_attn.linear_out.bias',
                   'encoder.encoders.6.self_attn.pos_bias_u',
                   'encoder.encoders.6.self_attn.pos_bias_v',
                ],
        "outs" : ['2017']},
        {"inps" : ['2116',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3376',            #   pos_emb weight
                   '3364','encoder.encoders.7.self_attn.linear_q.bias',
                   '3368','encoder.encoders.7.self_attn.linear_k.bias',
                   '3372','encoder.encoders.7.self_attn.linear_v.bias',
                   '3382','encoder.encoders.7.self_attn.linear_out.bias',
                   'encoder.encoders.7.self_attn.pos_bias_u',
                   'encoder.encoders.7.self_attn.pos_bias_v',
                ],
        "outs" : ['2227']},
        {"inps" : ['2326',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3399',            #   pos_emb weight
                   '3387','encoder.encoders.8.self_attn.linear_q.bias',
                   '3391','encoder.encoders.8.self_attn.linear_k.bias',
                   '3395','encoder.encoders.8.self_attn.linear_v.bias',
                   '3405','encoder.encoders.8.self_attn.linear_out.bias',
                   'encoder.encoders.8.self_attn.pos_bias_u',
                   'encoder.encoders.8.self_attn.pos_bias_v',
                ],
        "outs" : ['2437']},
        {"inps" : ['2536',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3422',            #   pos_emb weight
                   '3410','encoder.encoders.9.self_attn.linear_q.bias',
                   '3414','encoder.encoders.9.self_attn.linear_k.bias',
                   '3418','encoder.encoders.9.self_attn.linear_v.bias',
                   '3428','encoder.encoders.9.self_attn.linear_out.bias',
                   'encoder.encoders.9.self_attn.pos_bias_u',
                   'encoder.encoders.9.self_attn.pos_bias_v',
                ],
        "outs" : ['2647']},
        {"inps" : ['2746',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3445',            #   pos_emb weight
                   '3433','encoder.encoders.10.self_attn.linear_q.bias',
                   '3437','encoder.encoders.10.self_attn.linear_k.bias',
                   '3441','encoder.encoders.10.self_attn.linear_v.bias',
                   '3451','encoder.encoders.10.self_attn.linear_out.bias',
                   'encoder.encoders.10.self_attn.pos_bias_u',
                   'encoder.encoders.10.self_attn.pos_bias_v',
                ],
        "outs" : ['2857']},
        {"inps" : ['2956',         #   norm_mha_out
                   'speech_lengths',          #   mask
                   '603', '3468',            #   pos_emb weight
                   '3456','encoder.encoders.11.self_attn.linear_q.bias',
                   '3460','encoder.encoders.11.self_attn.linear_k.bias',
                   '3464','encoder.encoders.11.self_attn.linear_v.bias',
                   '3474','encoder.encoders.11.self_attn.linear_out.bias',
                   'encoder.encoders.11.self_attn.pos_bias_u',
                   'encoder.encoders.11.self_attn.pos_bias_v',
                ],
        "outs" : ['3067']},
    ]

cross_attn_nodes = [
        {"inps" : ["476",       #   q
                "214",          #   enc_in
                "encoder_out_lens",      #   mask
                "1793", "decoder.decoders.0.src_attn.linear_q.bias",
                "1797", "decoder.decoders.0.src_attn.linear_k.bias", 
                "1801", "decoder.decoders.0.src_attn.linear_v.bias",
                "1807", "decoder.decoders.0.src_attn.linear_out.bias",
                "decoder.decoders.0.norm2.weight",
                "decoder.decoders.0.norm2.bias",
                ],
        "outs" : ["575"]},
        {"inps" : ["693",       #   q
                "214",          #   enc_in
                "encoder_out_lens",      #   mask
                "1825", "decoder.decoders.1.src_attn.linear_q.bias",
                "1829", "decoder.decoders.1.src_attn.linear_k.bias", 
                "1833", "decoder.decoders.1.src_attn.linear_v.bias",
                "1839", "decoder.decoders.1.src_attn.linear_out.bias",
                "decoder.decoders.1.norm2.weight",
                "decoder.decoders.1.norm2.bias",
                ],
        "outs" : ["792"]},
        {"inps" : ["910",       #   q
                "214",          #   enc_in
                "encoder_out_lens",      #   mask
                "1857", "decoder.decoders.2.src_attn.linear_q.bias",
                "1861", "decoder.decoders.2.src_attn.linear_k.bias", 
                "1865", "decoder.decoders.2.src_attn.linear_v.bias",
                "1871", "decoder.decoders.2.src_attn.linear_out.bias",
                "decoder.decoders.2.norm2.weight",
                "decoder.decoders.2.norm2.bias",
                ],
        "outs" : ["1009"]},
        {"inps" : ["1127",       #   q
                "214",          #   enc_in
                "encoder_out_lens",      #   mask
                "1889", "decoder.decoders.3.src_attn.linear_q.bias",
                "1893", "decoder.decoders.3.src_attn.linear_k.bias", 
                "1897", "decoder.decoders.3.src_attn.linear_v.bias",
                "1903", "decoder.decoders.3.src_attn.linear_out.bias",
                "decoder.decoders.3.norm2.weight",
                "decoder.decoders.3.norm2.bias",
                ],
        "outs" : ["1226"]},
        {"inps" : ["1344",       #   q
                "214",          #   enc_in
                "encoder_out_lens",      #   mask
                "1921", "decoder.decoders.4.src_attn.linear_q.bias",
                "1925", "decoder.decoders.4.src_attn.linear_k.bias", 
                "1929", "decoder.decoders.4.src_attn.linear_v.bias",
                "1935", "decoder.decoders.4.src_attn.linear_out.bias",
                "decoder.decoders.4.norm2.weight",
                "decoder.decoders.4.norm2.bias",
                ],
        "outs" : ["1443"]},
        {"inps" : ["1561",       #   q
                "214",          #   enc_in
                "encoder_out_lens",      #   mask
                "1953", "decoder.decoders.5.src_attn.linear_q.bias",
                "1957", "decoder.decoders.5.src_attn.linear_k.bias", 
                "1961", "decoder.decoders.5.src_attn.linear_v.bias",
                "1967", "decoder.decoders.5.src_attn.linear_out.bias",
                "decoder.decoders.5.norm2.weight",
                "decoder.decoders.5.norm2.bias",
                ],
        "outs" : ["1660"]},
    ]

self_attn_nodes = [
        {"inps" : ["377",       #   q
                "377",          #   enc_in
                "hyps_lens_sos",      #   mask
                "1778", "decoder.decoders.0.self_attn.linear_q.bias",
                "1782", "decoder.decoders.0.self_attn.linear_k.bias", 
                "1786", "decoder.decoders.0.self_attn.linear_v.bias",
                "1792", "decoder.decoders.0.self_attn.linear_out.bias",
                "decoder.decoders.0.norm1.weight",
                "decoder.decoders.0.norm1.bias",
                ],
        "outs" : ["476"]},
        {"inps" : ["594",       #   q
                "594",          #   enc_in
                "hyps_lens_sos",      #   mask
                "1810", "decoder.decoders.1.self_attn.linear_q.bias",
                "1814", "decoder.decoders.1.self_attn.linear_k.bias", 
                "1818", "decoder.decoders.1.self_attn.linear_v.bias",
                "1824", "decoder.decoders.1.self_attn.linear_out.bias",
                "decoder.decoders.1.norm1.weight",
                "decoder.decoders.1.norm1.bias",
                ],
        "outs" : ["693"]},
        {"inps" : ["811",       #   q
                "811",          #   enc_in
                "hyps_lens_sos",      #   mask
                "1842", "decoder.decoders.2.self_attn.linear_q.bias",
                "1846", "decoder.decoders.2.self_attn.linear_k.bias", 
                "1850", "decoder.decoders.2.self_attn.linear_v.bias",
                "1856", "decoder.decoders.2.self_attn.linear_out.bias",
                "decoder.decoders.2.norm1.weight",
                "decoder.decoders.2.norm1.bias",
                ],
        "outs" : ["910"]},
        {"inps" : ["1028",       #   q
                "1028",          #   enc_in
                "hyps_lens_sos",      #   mask
                "1874", "decoder.decoders.3.self_attn.linear_q.bias",
                "1878", "decoder.decoders.3.self_attn.linear_k.bias", 
                "1882", "decoder.decoders.3.self_attn.linear_v.bias",
                "1888", "decoder.decoders.3.self_attn.linear_out.bias",
                "decoder.decoders.3.norm1.weight",
                "decoder.decoders.3.norm1.bias",
                ],
        "outs" : ["1127"]},
        {"inps" : ["1245",       #   q
                "1245",          #   enc_in
                "hyps_lens_sos",      #   mask
                "1906", "decoder.decoders.4.self_attn.linear_q.bias",
                "1910", "decoder.decoders.4.self_attn.linear_k.bias", 
                "1914", "decoder.decoders.4.self_attn.linear_v.bias",
                "1920", "decoder.decoders.4.self_attn.linear_out.bias",
                "decoder.decoders.4.norm1.weight",
                "decoder.decoders.4.norm1.bias",
                ],
        "outs" : ["1344"]},
        {"inps" : ["1462",       #   q
                "1462",          #   enc_in
                "hyps_lens_sos",      #   mask
                "1938", "decoder.decoders.5.self_attn.linear_q.bias",
                "1942", "decoder.decoders.5.self_attn.linear_k.bias", 
                "1946", "decoder.decoders.5.self_attn.linear_v.bias",
                "1952", "decoder.decoders.5.self_attn.linear_out.bias",
                "decoder.decoders.5.norm1.weight",
                "decoder.decoders.5.norm1.bias",
                ],
        "outs" : ["1561"]},
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process onnx file for trt engine generation')
    parser.add_argument('--input_onnx', type=str, required=True, help="input onnx model path")
    parser.add_argument('--streaming', action='store_true', help="using u2pp streaming onnx encoder input")
    parser.add_argument('--output_onnx', type=str, required=True, help="output onnx model path")

    args = parser.parse_args()

    graph = gs.import_onnx(onnx.load(args.input_onnx))

    if not args.streaming and 'encoder' in args.input_onnx:
      graph = addCast(graph)
      self_attn_nodes = enc_self_attn_nodes
      cross_attn_nodes = {}

    tmap = graph.tensors()
    
    for i,itn in enumerate(self_attn_nodes):
        inputs = [tmap[i] for i in itn["inps"]]
        outputs = [tmap[i] for i in itn["outs"]]
        name = "self_attn_{}".format(i)
        if 'encoder' in args.input_onnx:
                attrs = None
        else:
                attrs = {"AttentionType":"self"}
        graph.replace_attn(inputs, outputs, name, attrs)    

    for i,itn in enumerate(cross_attn_nodes):
        inputs = [tmap[i] for i in itn["inps"]]
        outputs = [tmap[i] for i in itn["outs"]]
        name = "cross_attn_{}".format(i)
        attrs = {"AttentionType":"cross"}
        graph.replace_attn(inputs, outputs, name, attrs)

    graph.cleanup().toposort()


    graph = replaceLayerNorm(graph)
    
    #if 'encoder' in args.input_onnx:
    #  graph = swpDiv(graph)
    
    onnx.save(gs.export_onnx(graph), args.output_onnx)
