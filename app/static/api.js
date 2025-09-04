export async function fbInfer({ path, axis, index, dt_us = null }) {
    const form = new FormData();
    form.append('path', path);
    form.append('axis', axis);
    form.append('index', index);
    if (dt_us !== null && dt_us !== undefined) {
        form.append('dt_us', dt_us);
    }
    const res = await fetch('/fbpick/infer', { method: 'POST', body: form });
    if (!res.ok) {
        throw new Error(await res.text());
    }
    const { cache_id, meta } = await res.json();
    return { cache_id, meta };
}

export async function fbPicks({
    cache_id,
    path,
    axis,
    index,
    dt_us,
    t0_us,
    method = 'argmax',
    median_kernel = 5,
    gaussian_sigma = null,
    sg_window = null,
    sg_poly = 2,
    conf_threshold = null,
    max_jump = null,
    save = true,
    layer = 'fb_auto',
}) {
    const form = new FormData();
    form.append('cache_id', cache_id);
    form.append('path', path);
    form.append('axis', axis);
    form.append('index', index);
    form.append('dt_us', dt_us);
    form.append('t0_us', t0_us);
    form.append('method', method);
    form.append('median_kernel', median_kernel);
    if (gaussian_sigma !== null && gaussian_sigma !== undefined) {
        form.append('gaussian_sigma', gaussian_sigma);
    }
    if (sg_window !== null && sg_window !== undefined) {
        form.append('sg_window', sg_window);
    }
    form.append('sg_poly', sg_poly);
    if (conf_threshold !== null && conf_threshold !== undefined) {
        form.append('conf_threshold', conf_threshold);
    }
    if (max_jump !== null && max_jump !== undefined) {
        form.append('max_jump', max_jump);
    }
    form.append('save', save);
    form.append('layer', layer);
    const res = await fetch('/fbpick/picks', { method: 'POST', body: form });
    if (!res.ok) {
        throw new Error(await res.text());
    }
    const { picks, aux } = await res.json();
    return { picks, aux };
}

