import argparse
import threading

import argconf
import cherrypy
import torch

import deeplm


@cherrypy.expose
class TokenizationServer(object):

    def __init__(self, vocab, model):
        self.lock = threading.Lock()
        self.model = model
        self.vocab = vocab

    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def POST(self):
        text = cherrypy.request.json["text"]
        inp = [[self.vocab.tok2idx[tok] for tok in text]]
        inp = torch.Tensor(inp).long().cuda()
        length = [len(text)]
        length = torch.Tensor(length).long().cuda()
        with self.lock:
            scores = self.model(inp, length)
        output = scores.max(1)[1].view(-1).cpu().numpy().tolist()
        final_output = []
        decode_idx = 0
        for o in output:
            final_output.append(text[decode_idx])
            if o == 1:
                final_output.append("-")
            decode_idx += 1
        return dict(text="".join(final_output))


def main():
    parser = argparse.ArgumentParser(description="Runs the server for tokenization as a service.", 
        epilog="Usage:\npython -m trident.utils.run_server -c confs/default.json")
    parser.add_argument("--config", "-c", type=str, default="confs/default.json")
    parser.add_argument("--port", "-p", type=int, default=8080)
    args = parser.parse_args()

    config = argconf.config_from_json(args.config)
    vocab, _ = deeplm.data.Seq2SeqDataset.iters(config)
    model = deeplm.model.InterstitialModel(vocab, config)
    sd = torch.load(config["resume"])
    model.load_state_dict(sd["state"])
    model.avg_param = sd["ema"]
    model.steps_ema = sd["steps_ema"]
    model.cuda()
    model.eval()
    model.load_ema_params()

    cherrypy.config.update({"global": {"engine.autoreload.on": False}})
    cherrypy.quickstart(TokenizationServer(vocab, model), "/", {"/": {"request.dispatch": cherrypy.dispatch.MethodDispatcher()}})


if __name__ == "__main__":
    main()