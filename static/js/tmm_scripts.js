  
    let linePlot; // Reference to the Chart.js line plot
    let electricFieldData = [];
    
    
    document.addEventListener('DOMContentLoaded', function () {
      // Animate cards on page load
      gsap.from('.card', {
        duration: 1,
        y: 50,
        opacity: 0,
        stagger: 0.2,
        ease: 'power3.out'
      });
    
      // Animate the header
      gsap.from('h1', {
        duration: 1,
        y: -50,
        opacity: 0,
        ease: 'bounce.out'
      });
    });

  // Function to add a new row at a specific position
function addRow() {
    
    const table = document.getElementById('layerTable').getElementsByTagName('tbody')[0];
    const positionInput = document.getElementById('insertPosition');
    const position = parseInt(positionInput.value);

    // Validate the position
    if (!position || position < 1 || position > table.rows.length + 1) {
      alert(`Invalid position. Please enter a value between 1 and ${table.rows.length + 1}.`);
      return;
    }

    // Insert the row at the specified position
    const row = table.insertRow(position - 1); // Convert 1-based index to 0-based
    row.innerHTML = `
      <td>${position}</td>
      <td><input type="number" step="0.01" class="form-control" name="refractive_index" required></td>
      <td><input type="number" step="0.01" class="form-control" name="thickness" required></td>
      <td><button class="btn btn-danger btn-sm" onclick="removeRow(this)">Remove</button></td>
      `;
    // Update the serial numbers of all subsequent rows
    updateRowNumbers();
}

  // Function to remove a row
  function removeRow(button) {
    const row = button.parentNode.parentNode;
    row.parentNode.removeChild(row);
    updateRowNumbers(); // Update serial numbers after removal
  }

  // Function to update row numbers
  function updateRowNumbers() {
    const table = document.getElementById('layerTable').getElementsByTagName('tbody')[0];
    for (let i = 0; i < table.rows.length; i++) {
      table.rows[i].cells[0].textContent = i + 1; // Update Sr. No.
    }
  }

  // Function to submit data

  function submitData() {
  
    const layers = [];
    const table = document.getElementById('layerTable').getElementsByTagName('tbody')[0];
    for (let row of table.rows) {
      const refractiveIndex = row.cells[1].getElementsByTagName('input')[0].value;
      const thickness = row.cells[2].getElementsByTagName('input')[0].value;
      layers.push({ refractiveIndex: parseFloat(refractiveIndex), thickness: parseFloat(thickness) });
    }

    const wavelengthStart = parseFloat(document.getElementById('wavelengthStart').value);
    const wavelengthEnd = parseFloat(document.getElementById('wavelengthEnd').value);
    const stepPoint = parseFloat(document.getElementById('stepPoint').value);


    const data = { layers, wavelengthStart, wavelengthEnd, stepPoint };

    fetch('/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(result => {
                
        const slider = document.getElementById("electricFieldSlider");
        slider.max = result.wavelengths.length-1; // Set slider max to numPoints

        const sliderVal = document.getElementById("sliderValue");
        sliderVal.innerText=  result.wavelengths[0]      // set slider value
        
        // Print the electric field data in the console
        //console.log("Refractive index Data:", result.layerProfile);
        //console.log("x values Data:", result.x_values);
        plotReflectance(result.wavelengths, result.reflectance);
        //plotindex(result.x_values,result.layerProfile);
        E2_heatmap2(result.x_values,result.wavelengths,result.electricIntensity);

        plotCombinedData(result.layerProfile, result.electricIntensity,result.x_values,result.wavelengths) ;
        
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }

  function plotReflectance(wavelengths, reflectance) {
    //System.out.println(reflectance);
    const ctx = document.getElementById('reflectancePlot').getContext('2d');
    //console.log("Wavelengths are: ",wavelengths);
    if (window.reflectanceChart) {
      window.reflectanceChart.destroy();
    }

      // Convert wavelengths and reflectance into {x, y} format
      const dataPoints = wavelengths.map((wavelength, index) => ({
        x: parseFloat(wavelength),
        y: parseFloat(reflectance[index])
      }));
  
    window.reflectanceChart = new Chart(ctx, {
      type: 'line',
      data: {
        //labels: wavelengths,
        datasets: [{
          label: 'Reflectance (%)',
          data: dataPoints,
          borderColor: 'blue',
          borderWidth: 1,
          fill: false,
          pointRadius: 5, // Adjust scatter point size
          pointStyle: 'triangle', // Change scatter point type 'rect','rectRot',cross, crossrot

        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: false // Hide legend
          }
        },
        scales: {
          
          x: {
            type: 'linear', // Ensure numerical scale
            title: { display: true, text: 'Wavelength (nm)'},
            ticks: {
            callback: (value) => parseFloat(value).toFixed(0) // Limit decimal places
        }
        },
          y: { title: { display: true, text: 'Reflectance (%)' }}
        }
      }
    });
  }


  // Heatmap Data and Configuration
  function E2_heatmap2(x_val,wavelengths,electricIntensity){
      
    const trace = {
      z: electricIntensity,
      x: x_val, // x-axis values (length)
      y: wavelengths, // y-axis values (wavelengths)
      type: 'heatmap',
      colorscale: 'Viridis', // Color scheme for the heatmap
      colorbar: {
        title: 'Intensity',
        titleside: 'right',
        tickformat: '.2f' // Set to 2 decimal places (you can change this value as needed)

      }
    };


    const layout = {
      autosize: true,  // Ensures the heatmap resizes properly
      //height: 600, // Adjust this value to fit your container
      margin: { l: 60, r: 25, t: 25, b: 60 }, // Reduce margins to maximize plot area
      //title: 'Heatmap of Electric Field Intensity',
      xaxis: { title: 'Z (nm)' },
      yaxis: { title: 'Wavelength (nm)' },
    };

    Plotly.newPlot('heatmapPlot2', [trace], layout);

  }

  //--------------- combined plot------------------------
 
  // Function to generate combined plot
function plotCombinedData(refractiveIndices, electricFieldIntensities, thicknesses,wavelengths) {
  const ctx = document.getElementById("indexPlot").getContext("2d");

  // Store electric field and wavelength data globally for updates
  window.electricFieldData=electricFieldIntensities;
  window.wavelengthData=wavelengths;// Store the wavelength data

  //const labels = thicknesses.map((_, i) => `Layer ${i + 1}`); // Use thickness values or layer indices as labels
  //console.log("Intensity Data:::", electricFieldIntensities);

  if (window.indexFieldChart) {
    window.indexFieldChart.destroy();
  }

// initialize the chart
  window.indexFieldChart = new Chart(ctx, {
    type: "line",
    data: {
      //labels: labels,
      labels:  thicknesses,
      datasets: [
        {
          label: "Refractive Index",
          data: refractiveIndices,
          borderColor: "blue",
          backgroundColor: "rgba(0, 0, 255, 0.2)",
          yAxisID: "y-axis-1",
          pointRadius: 0, // Remove scatter points
          borderWidth: 1,
        },
        {
          label: `Electric Field Intensity at ${wavelengths[0].toFixed(2)} nm`,
          data: electricFieldIntensities[0],// [0]//
          borderColor: "red",
          backgroundColor: "rgba(255, 0, 0, 0.2)",
          yAxisID: "y-axis-2",
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          title: {
            display: true,
            text: "Z (nm)",
            
          },
        },
        "y-axis-1": {
          type: "linear",
          position: "left",
          title: {
            display: true,
            text: "Refractive Index",
          },
        },
        "y-axis-2": {
          type: "linear",
          ticks: {
            callback: function(value) {
                return parseFloat(value).toFixed(1); // Set two decimal places
            } },
          position: "right",
          title: {
            display: true,
            text: "Electric Field Intensity",
          },
          grid: {
            drawOnChartArea: false, // Avoid overlapping grid lines
          },
        },
      },
    },
  });
}

/*
// Example Data for Testing
const refractiveIndices = [1.5, 2.3, 1.8, 2.1]; // Replace with your dynamic data
const electricFieldIntensities = [0.8, 1.2, 1.1, 1.5]; // Replace with your dynamic data
const thicknesses = [100, 200, 150, 250]; // Layer thicknesses
const wavelengths = [400,550,600];

// Call the function to plot combined data
plotCombinedData(refractiveIndices, electricFieldIntensities, thicknesses,wavelengths);
*/

// updating electric field----------------
// Function to update electric field intensity based on wavelength index
function updateElectricFieldDataset(index) {
  document.getElementById("sliderValue").innerText = index;
  if (!window.electricFieldData || index >= window.electricFieldData.length) {
    console.error("Invalid index for electric field data update.");
    return;
  }

  // Update electric field dataset
  const newIntensityData = window.electricFieldData[index];
  const newLabel = `Electric Field Intensity at ${window.wavelengthData[index].toFixed(2)} nm`;
  
  const sliderVal = document.getElementById("sliderValue");
  sliderVal.innerText=  window.wavelengthData[index].toFixed(2)      // set slider value

  //console.log("new intensity dataset:", newIntensityData );

  const electricFieldDataset = window.indexFieldChart.data.datasets[1]; //[1] is index is for electric field and we only updates it
  electricFieldDataset.data = newIntensityData;
  electricFieldDataset.label = newLabel;
  
  // Update the chart
  window.indexFieldChart.update();
}


// Function to get a random color for each line (for better distinction in the legend)
function getRandomColor() {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}
