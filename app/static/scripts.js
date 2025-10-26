// Menu Sidebar 
function openNav() {
    document.getElementById("mySidebar").style.width = "250px";
    document.body.style.backgroundColor = "rgba(0,0,0,0.4)";
}

function closeNav() {
    document.getElementById("mySidebar").style.width = "0";
    document.body.style.backgroundColor = "white";
}


// History Sidebar
  const panel = document.getElementById('sidePanel');
  const togglePanelBtn = document.getElementById('togglePanelBtn');
  const sideText = document.getElementById('sideText');
  const main = document.getElementById('cameraMain');
  const pageTitle2 = document.getElementById('cameraTitle')

  togglePanelBtn.addEventListener('click', () => {
    panel.classList.toggle('collapsed');
    togglePanelBtn.textContent = panel.classList.contains('collapsed') ? '⮞' : '⮜';
    if (!panel.classList.contains('collapsed')){
        main.style.marginRight = "250px";
    }else{ main.style.marginRight = "0px";}
    pageTitle2.style.display = "none";
  });