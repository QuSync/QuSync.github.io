

// Function to check if containers are in the viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight)
    );
}

// Function to check if containers are in the viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight)
    );
}

// Function to handle scroll events
function handleScroll() {
    const containers = document.querySelectorAll('.container');
    containers.forEach(container => {
        if (isInViewport(container)) {
            container.classList.add('visible');
        } else {
            container.classList.remove('visible');
        }
    });
}

// Add scroll event listener
window.addEventListener('scroll', handleScroll);

// Trigger handleScroll on page load to check initial visibility
handleScroll();

function scrollToBottom() {
    
document.getElementById("bottom").scrollIntoView({behavior:"smooth"});
}