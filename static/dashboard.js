const $ = sel => document.querySelector(sel);
let currentTrain = null;
let engines = new Set();

function log(obj) {
  const pre = $("#log");
  pre.textContent = JSON.stringify(obj, null, 2) + "\n" + pre.textContent;
}

function setTotals(t) {
  $("#loc").textContent = t.locomotive || 0;
  $("#cars").textContent = t.railcar || 0;
}

async function initCharts() {
  // Daily summary: EB/WB
  const res = await fetch("/api/summary/daily");
  const daily = (await res.json()).data;
  const labels = daily.map(d => d.day);
  const trainsEB = daily.map(d => d.trains_EB || 0);
  const trainsWB = daily.map(d => d.trains_WB || 0);
  const carsEB = daily.map(d => d.cars_EB || 0);
  const carsWB = daily.map(d => d.cars_WB || 0);

  // Trains per day, stacked EB/WB
  const ctx1 = document.getElementById('dailyChart');
  new Chart(ctx1, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Trains EB', data: trainsEB, backgroundColor: '#4e79a7', stack: 'tr' },
        { label: 'Trains WB', data: trainsWB, backgroundColor: '#a0cbe8', stack: 'tr' },
        { label: 'Cars EB', data: carsEB, backgroundColor: '#f28e2b66', stack: 'car' },
        { label: 'Cars WB', data: carsWB, backgroundColor: '#f7b6d2', stack: 'car' }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } },
      scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } }
    }
  });

  // Cars per Train (recent) colored by direction
  const res2 = await fetch("/api/trains/recent");
  const rec = (await res2.json()).data;
  const lab2 = rec.map(r => r.train_id);
  const cars2 = rec.map(r => r.railcars);
  const color2 = rec.map(r => r.direction === 'EB' ? '#59a14f' : '#e15759');
  const ctx2 = document.getElementById('cptChart');
  new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: lab2,
      datasets: [{
        label: 'Cars/train',
        data: cars2,
        backgroundColor: color2
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } },
      scales: { x: { display: false }, y: { beginAtZero: true } }
    }
  });
}

function refreshEnginesUI() {
  $("#engines").textContent = Array.from(engines).sort().join(", ") || "—";
}

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = ev => {
    const data = JSON.parse(ev.data);
    log(data);
    if (data.event === "train_start") {
      currentTrain = data.train_id;
      engines.clear();
      $("#activeTrain").textContent = currentTrain;
      $("#direction").textContent = "—";
      $("#speed").textContent = "—";
      setTotals({});
      refreshEnginesUI();
    }
    if (data.event === "count") {
      setTotals(data.totals || {});
      if (data.speed_mph !== undefined) {
        $("#speed").textContent = data.speed_mph;
      }
    }
    if (data.event === "engine_number") {
      if (data.train_id === currentTrain && data.engine_number) {
        engines.add(data.engine_number);
        refreshEnginesUI();
      }
    }
    if (data.event === "train_end") {
      if (data.train_id === currentTrain) {
        $("#direction").textContent = data.direction || "—";
        $("#speed").textContent = data.avg_speed_mph || "—";
      }
      currentTrain = null;
      // You could also trigger a refresh of charts here
    }
  };
}

initCharts();
connectWS();
