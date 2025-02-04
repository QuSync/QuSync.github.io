// Tab functionality
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and tab contents
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(tc => tc.classList.remove('active'));

        // Add active class to clicked tab and corresponding content
        tab.classList.add('active');
        document.getElementById(tab.dataset.target).classList.add('active');
    });
});

// Add conversion logic for each form
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const inputUnit = form.querySelector('select').value;
        const inputValue = form.querySelector('input').value;
        const resultsDiv = form.nextElementSibling;

        const response = await fetch("/convertt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input_unit: inputUnit, input_value: inputValue }),
        });

        const data = await response.json();

        if (response.ok) {
            resultsDiv.innerHTML = Object.entries(data)
                .map(([unit, value]) => `<p>${unit}: ${value.toFixed(6)}</p>`)
                .join("");
        } else {
            resultsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
        }
    });
});

//-------Ensure to remove the spinner when the page load------------------
//document.addEventListener("DOMContentLoaded", function () {
//   document.getElementById("spinner").classList.remove("show");
//});

