# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import time
 
class AudioDetokenizerModel:

    def __init__(self,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 lora_config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow = flow
        self.hift = hift
        self.dtype = torch.float16
        # self.dtype = torch.bfloat16
        self.max_seq_short = 384
        self.max_seq_long = 2048
        self.max_batch = 1

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval().to(self.dtype)
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def inference(self, vp_emb, tts_speech_token,
                  prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32), is_en=False, **kwargs):

        torch.cuda.synchronize()
        t0 = time.time()

        torch.cuda.synchronize()
        t1 = time.time()
        
        tts_mel = self.flow.inference(token=tts_speech_token.to(self.device),
                                      token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                                      prompt_token=prompt_speech_token.to(self.device),
                                      prompt_token_len=prompt_speech_token_len.to(self.device),
                                      prompt_feat=prompt_speech_feat.to(self.device),
                                      prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                      embedding=vp_emb.to(self.device).to(self.dtype)).float()
        torch.cuda.synchronize()

        tts_speech = self.hift.inference(mel=tts_mel).cpu()
        torch.cuda.synchronize()
        dur = tts_speech.shape[-1]/22050
        torch.cuda.empty_cache()
        return {'tts_speech': tts_speech}
