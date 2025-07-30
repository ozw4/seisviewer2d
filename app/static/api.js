async function fetchAndPlot(key1) {
    const res = await fetch(`/get_section?key1=${key1}`);
    const data = await res.json();
    plotSeismicData(data);
}
