$(document).ready(function () {
    let typingTimer; // Timer for debouncing
    const debounceDelay = 100; // Delay in ms

    // Function to update the plot
    function updatePlot(func) {
        $.ajax({
            type: 'POST',
            url: '/update_plot',
            contentType: 'application/json',
            data: JSON.stringify({ function: func }),
            success: function (response) {
                if (response.error) {
                    console.warn("Error:", response.error); // Log errors gracefully
                } else {
                    Plotly.react('plot', [{
                        x: response.x,
                        y: response.y,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: 'blue' }
                    }]);
                }
            },
            error: function (err) {
                console.error("AJAX error:", err);
            }
        });
    }

    // Trigger the plot update on input
    $('#function').on('keyup', function () {
        const func = $(this).val();
        clearTimeout(typingTimer); // Clear the previous timer
        typingTimer = setTimeout(() => {
            if (func.trim() !== '') {
                updatePlot(func);
            }
        }, debounceDelay); // Add debounce delay
    });

    // Initialize an empty plot
    Plotly.newPlot('plot', [{
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines+markers'
    }], {
        title: 'Real-Time Interactive Plot',
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });
});