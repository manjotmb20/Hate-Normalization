from flask import Flask, request, render_template
from regression import model1,tokenizer_new,tokenize_new
import numpy as np

from transformers import BartForConditionalGeneration, BartTokenizer,BartConfig
import torch
config = BartConfig.from_json_file('output_model/hate/config.json')
model=BartForConditionalGeneration.from_pretrained('output_model/hate/')
tok = BartTokenizer.from_pretrained('output_model/hate/')

app = Flask(__name__)
app.debug = True


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        name = request.form["name"]
        hate=" "
        if(len(name)>0):
            if name.split(" ")[-1]=='':
                a,b,c=tokenize_new([name],tokenizer_new)
                out=np.round(model1.predict([a,b])[0][0])
                if out<=3:
                    hate="No Hate detected"    
                elif out>3 and out<=5:
                    hate="LOW"
                elif out>5 and out<=7:
                    hate="MEDIUM"    
                else:
                    hate="HIGH"    
                batch = tok(name, return_tensors='pt')
                generated_ids = model.generate(batch['input_ids'])
                name=tok.batch_decode(generated_ids, skip_special_tokens=True)
        return "Hate Strength detected: "+"|"+hate+"|" + name[0]
    return render_template("test.html")


if __name__ == "__main__":
    app.run()