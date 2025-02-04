from flask import Flask, render_template, request, jsonify
import os
from utils.unit_conversion import convert_units
from utils.plotting import evaluate_function
import numpy as np
from utils.tmm import tmm


app = Flask(__name__)

@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route('/api/data')
def get_data():
    return jsonify({"message": "Hello from Flask!"})

@app.route("/unit_conversion.html")
@app.route("/unit_conversion")
def uc():
    return render_template("unit_conversion.html")

@app.route("/tmm.html")
@app.route("/tmm")
def tm_method():
    return render_template("tmm.html")

@app.route("/about.html")
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact.html")
@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/convertt", methods=["POST"])
def convert():
    data = request.json
    input_unit = data.get("input_unit")
    input_value = float(data.get("input_value", 0))

    results = convert_units(input_unit, input_value)
    
    if results is None:
        return jsonify({"error": "Invalid input unit"}), 400
    
    return jsonify(results)

# Ensure the plots folder exists
os.makedirs("static/plots", exist_ok=True)

@app.route("/graph_plot.html")
def g_plot():
    return render_template("graph_plot.html")

@app.route("/update_plot", methods=["POST"])
def update_plot():
    function = request.json.get("function", "")
    result = evaluate_function(function)
    return jsonify(result)


@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    layers = data['layers']  # see javascript >>  function submitData() where these strings are defined....
    wavelength_start = data['wavelengthStart']
    wavelength_end = data['wavelengthEnd']
    num_step = data['stepPoint']
    #print( "wavelength step size is:",num_step)
            # Debugging: log incoming data
    #print(f"Received data: {data}")


    fulln = [float (layer['refractiveIndex']) for layer in layers]
    fullw = [float (layer['thickness']) for layer in layers]
    fulln = np.insert([fulln[0]],1,fulln)
    fullw = np.insert([0],1,fullw) #starting from 0 (origin)
    fullw=np.array(fullw)
    print("fullw is::",fullw)
    print("fulln is::",fulln)
    
    #wvls = np.linspace(wavelength_start, wavelength_end, num_points)
    wvls = np.arange(wavelength_start, wavelength_end+num_step, num_step)
    #print(wvls)
    #wvls = np.linspace(400, 1000, 601)
    R = []
    E_field=[]
    E_intensity=[]
    for wvl in wvls:
        #print(wvl)
        r, t, x, Nn, E  = tmm(wvl, fulln, fullw)
        R.append(abs(r)**2 * 100)
        #E_field.append(E) # JSON doesnot support complex thus separated both
                # Convert complex E field to real and imaginary parts
                
        E_field.append({
            "real": E.real.tolist(),
            "imaginary": E.imag.tolist()
        })

        E_intensity.append(np.abs(E**2))

    E_field = np.array(E_field)
    E_intensity=np.array(E_intensity)

    # Return results
    return jsonify({'wavelengths': wvls.tolist(), 'reflectance': R,'electricField': E_field.tolist() ,'electricIntensity': E_intensity.tolist(),
                   'x_values':x.tolist(),'layerProfile': Nn.tolist()})



if __name__ == "__main__":
    app.run(debug=True)
