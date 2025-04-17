// Main JavaScript file for the Model Preference Testing application

// Theme handling
function setTheme(themeName) {
    // Delegate to ThemeManager
    ThemeManager.applyTheme(themeName);
}

function toggleTheme() {
    const currentTheme = localStorage.getItem('theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

// Theme state management
const ThemeManager = {
    callbacks: [],
    
    // Register callbacks to run when theme changes
    onThemeChange(callback) {
        this.callbacks.push(callback);
    },
    
    // Notify all registered callbacks
    notifyCallbacks(themeName) {
        this.callbacks.forEach(callback => {
            try {
                callback(themeName);
            } catch (e) {
                console.error('Theme change callback error:', e);
            }
        });
    },
    
    // Apply theme with notification
    applyTheme(themeName) {
        // Store theme preference
        localStorage.setItem('theme', themeName);
        
        // Apply to both html and body
        document.documentElement.setAttribute('data-theme', themeName);
        document.body.setAttribute('data-theme', themeName);
        
        // Force a repaint to ensure all CSS variables are applied immediately
        document.body.style.display = 'none';
        document.body.offsetHeight; // Trigger a reflow
        document.body.style.display = '';
        
        // Notify callbacks of theme change
        this.notifyCallbacks(themeName);
    }
};

// Initialize theme
function initTheme() {
    // Check for saved theme preference or set default to dark
    const savedTheme = localStorage.getItem('theme') || 'dark';
    
    // This creates a consistent state between our variable store and the DOM
    ThemeManager.applyTheme(savedTheme);
    
    // Register any global callbacks that should happen on theme change
    ThemeManager.onThemeChange((theme) => {
        // Update chart colors if chart exists and needs rerendering
        if (window.resultsChart) {
            window.resultsChart.update();
        }
    });
}

// Handle theme toggle click
function handleThemeToggle() {
    const themeMenu = document.querySelector('.theme-menu');
    themeMenu.classList.toggle('show');
}

// Theme UI initialization
function initializeThemeToggle() {
    // Only create theme toggle elements if they don't exist
    if (document.querySelector('.theme-toggle-container')) {
        return; // Already initialized
    }
    
    // Create theme toggle container
    const themeToggleContainer = document.createElement('div');
    themeToggleContainer.className = 'theme-toggle-container';
    
    // Create the gear icon toggle button
    const themeToggle = document.createElement('button');
    themeToggle.className = 'theme-toggle';
    themeToggle.innerHTML = '<i class="bi bi-gear"></i>';
    themeToggle.setAttribute('aria-label', 'Theme settings');
    themeToggle.addEventListener('click', handleThemeToggle);
    
    // Create the theme menu
    const themeMenu = document.createElement('div');
    themeMenu.className = 'theme-menu';
    
    // Add theme options with correct active states
    const currentTheme = localStorage.getItem('theme') || 'dark';
    themeMenu.innerHTML = `
        <div class="theme-menu-item" onclick="setTheme('dark')">
            <i class="bi ${currentTheme === 'dark' ? 'bi-check2 text-primary' : 'bi-moon-fill'}"></i> Dark Mode
        </div>
        <div class="theme-menu-item" onclick="setTheme('light')">
            <i class="bi ${currentTheme === 'light' ? 'bi-check2 text-primary' : 'bi-sun-fill'}"></i> Light Mode
        </div>
    `;
    
    // Add to container
    themeToggleContainer.appendChild(themeToggle);
    themeToggleContainer.appendChild(themeMenu);
    
    // Add to body
    document.body.appendChild(themeToggleContainer);
    
    // Setup click-outside handler
    document.addEventListener('click', function(event) {
        const themeMenu = document.querySelector('.theme-menu');
        const themeToggle = document.querySelector('.theme-toggle');
        
        if (themeMenu && themeToggle) {
            if (!themeMenu.contains(event.target) && !themeToggle.contains(event.target)) {
                themeMenu.classList.remove('show');
            }
        }
    });
}

// Main initialization when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme system
    initTheme();
    
    // Initialize theme toggle UI
    initializeThemeToggle();
    
    // Handle model submission form
    const modelForm = document.querySelector('form[action*="/submit"]');
    if (modelForm) {
        modelForm.addEventListener('submit', function(e) {
            // Store model name for retrieval in processing page
            const modelName = document.getElementById('model_name').value;
            sessionStorage.setItem('model_name', modelName);
        });
    }
    
    // Enable Bootstrap tooltips everywhere
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Enable Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});

// Removed the beforeunload event listener to allow users to navigate away freely