// ============================================================
        // DATA LOADING
        // ============================================================
        
        let summaryData = null;
        let trajectoryData = null;
        let countryData = null;
        let cumulativeCountryData = null;
        let gdpImpactData = null;
        let modelCurves = null;
        let methodsEnvData = null;
        let siteManifest = null;
        let siteGridResolutionDeg = null;
        let currentMapColorScale = null;
        let currentSiteData = null;
        let currentSiteGeojson = null;
        let map = null;
        let isInitialMapLoad = true;
        let siteLayer = null;
        let siteGridLayer = null;
        let siteTourismLayer = null;
        let siteGridRenderer = null;
        let siteTourismRenderer = null;
        let choroplethLayer = null;
        let countryBoundaries = null;
        let renderTimeout = null;  // For debouncing
        let isRendering = false;  // Prevent concurrent renders
        let useVectorTilesForSites = false;
        const siteFileCache = new Map();
        const VALUE_TYPES = ['tourism', 'coastal_protection', 'fisheries'];
        const VALUE_TYPE_LABELS = {
            all: 'All datasets',
            tourism: 'Tourism',
            fisheries: 'Fisheries',
            coastal_protection: 'Coastal protection',
        };
        const VALUE_TYPE_DESCRIPTIONS = {
            tourism: 'Reef-associated tourism value layer.',
            fisheries: 'Reef fisheries value represented at site points.',
            coastal_protection: 'Coastal protection value at shoreline points (zoom in to load).',
        };

        const HABITAT_ILLUSTRATION_OPTIONS = [0, 0.25, 0.4, 0.5, 0.75, 1];
        const MODEL_SORT_ORDER = ['Linear', 'Compound', 'Tipping Point'];

        let CHART_COLORS = {};

        function cssColor(varName, fallback) {
            const value = getComputedStyle(document.documentElement)
                .getPropertyValue(varName)
                .trim();
            return value || fallback;
        }

        function initChartColors() {
            CHART_COLORS = {
                dataset: {
                    tourism: cssColor('--dataset-tourism', '#3A9AB2'),
                    fisheries: cssColor('--dataset-fisheries', '#22c55e'),
                    coastal_protection: cssColor('--dataset-coastal', '#a78bfa'),
                },
                model: {
                    linear: cssColor('--model-linear', '#E3B710'),
                    compound: cssColor('--model-compound', '#F11B00'),
                    tipping: cssColor('--model-tipping', '#dc2626'),
                },
                scenario: {
                    rcp45: cssColor('--scenario-rcp45', '#3498db'),
                    rcp85: cssColor('--scenario-rcp85', '#e74c3c'),
                },
                lossScale: [
                    cssColor('--loss-low', '#22c55e'),
                    cssColor('--loss-mid', '#eab308'),
                    cssColor('--loss-high', '#E3B710'),
                    cssColor('--loss-extreme', '#F11B00'),
                ],
                habitatExportA: Number(cssColor('--habitat-export-a', '0.4')) || 0.4,
            };
        }
        
        let APP_BASE_PATH = '';
        let DATA_PATH = 'exported_data/';
        // Increment DATA_VERSION whenever exported_data/ files are regenerated to
        // prevent browsers serving stale JSON from the HTTP cache.
        const DATA_VERSION = '10';

        const PAGE_IDS = ['overview', 'map', 'trajectories', 'gdp', 'methods'];
        const PAGE_TITLES = {
            overview: 'Overview',
            map: 'Map',
            trajectories: 'Trajectories',
            gdp: 'Country-level Impact',
            methods: 'Methods',
        };
        const DEFAULT_PAGE = 'overview';
        const POINT_RADIUS_CONFIG = {
            lowZoomMax: 2,  // max zoom for low radius
            midZoomMax: 7,  // max zoom for mid radius
            low: 1,  // low radius
            mid: 2,  // mid radius
            high: 4,  // high radius
        };
        // Display-only alignment tweak for point layers.
        // Set small values (e.g., +/-0.0002) only if you observe a stable offset.
        const POINT_ALIGNMENT_OFFSET = {
            lat: 0.0,
            lng: 0.0,
        };

        function getPointRadius(zoom) {
            if (zoom < POINT_RADIUS_CONFIG.lowZoomMax) return POINT_RADIUS_CONFIG.low;
            if (zoom < POINT_RADIUS_CONFIG.midZoomMax) return POINT_RADIUS_CONFIG.mid;
            return POINT_RADIUS_CONFIG.high;
        }

        function applyPointAlignmentOffset(latlng) {
            if (!latlng) return latlng;
            const dLat = Number(POINT_ALIGNMENT_OFFSET.lat || 0);
            const dLng = Number(POINT_ALIGNMENT_OFFSET.lng || 0);
            if (dLat === 0 && dLng === 0) return latlng;
            return L.latLng(latlng.lat + dLat, latlng.lng + dLng);
        }

        function lonLatToTile(lon, lat, zoom) {
            const n = Math.pow(2, zoom);
            const clampedLat = Math.max(Math.min(lat, 85.05112878), -85.05112878);
            const x = Math.max(
                0,
                Math.min(
                    n - 1,
                    Math.floor(((lon + 180) / 360) * n)
                )
            );
            const latRad = (clampedLat * Math.PI) / 180;
            const y = Math.max(
                0,
                Math.min(
                    n - 1,
                    Math.floor(
                        ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n
                    )
                )
            );
            return { x, y };
        }

        function getTileKeysForBounds(bounds, zoom) {
            if (!bounds) return [];
            const north = bounds.getNorth();
            const south = bounds.getSouth();
            const west = bounds.getWest();
            const east = bounds.getEast();
            const n = Math.pow(2, zoom);

            const yMin = lonLatToTile(0, north, zoom).y;
            const yMax = lonLatToTile(0, south, zoom).y;

            const lonRanges = west <= east
                ? [{ west, east }]
                : [{ west, east: 180 }, { west: -180, east }];

            const keys = new Set();
            lonRanges.forEach((range) => {
                const xMin = lonLatToTile(range.west, 0, zoom).x;
                const xMax = lonLatToTile(range.east, 0, zoom).x;
                for (let x = xMin; x <= xMax; x++) {
                    const wrappedX = ((x % n) + n) % n;
                    for (let y = yMin; y <= yMax; y++) {
                        if (y >= 0 && y < n) {
                            keys.add(`${zoom}/${wrappedX}/${y}`);
                        }
                    }
                }
            });
            return Array.from(keys);
        }

        async function fetchGeoJsonWithCache(filename) {
            const cacheKey = `${filename}?v=${DATA_VERSION}`;
            if (siteFileCache.has(cacheKey)) {
                return siteFileCache.get(cacheKey);
            }
            const response = await fetch(`${DATA_PATH}${filename}?v=${DATA_VERSION}`);
            if (!response.ok) {
                return null;
            }
            const parsed = await response.json();
            siteFileCache.set(cacheKey, parsed);
            return parsed;
        }

        /**
         * Load pre-aggregated grid-cell features for the new compact export format.
         * Geometry is stored once per value_type; metrics are columnar arrays per scenario.
         */
        async function loadGriddedSiteFeatures({
            datasetKey,
            scenarioKey,
            model,
            scenarioDatasetKey,
            griddedEntry,
        }) {
            if (!griddedEntry?.grid_file || !griddedEntry?.metrics_file) {
                return [];
            }

            const [gridData, metricsData] = await Promise.all([
                fetchGeoJsonWithCache(griddedEntry.grid_file),
                fetchGeoJsonWithCache(griddedEntry.metrics_file),
            ]);

            const isPolygonGeojson =
                gridData?.type === 'FeatureCollection' && gridData?.geom_type === 'polygon';
            const hasGridCells = Array.isArray(gridData?.cells) && gridData.cells.length > 0;

            if (!isPolygonGeojson && !hasGridCells) {
                console.warn('Gridded data missing for', datasetKey, griddedEntry);
                return [];
            }
            if (!metricsData?.scenarios) {
                console.warn('Metrics data missing for', datasetKey);
                return [];
            }

            const metricsScenarios = metricsData.scenarios;
            const resolvedKey =
                resolveMetricsScenarioKey(metricsScenarios, datasetKey, scenarioKey, model) ||
                (metricsScenarios[scenarioDatasetKey] ? scenarioDatasetKey : null) ||
                Object.keys(metricsScenarios).find(
                    (key) => key.toLowerCase() === String(scenarioDatasetKey).toLowerCase()
                ) ||
                null;
            const scenarioMetrics = resolvedKey ? metricsScenarios[resolvedKey] : null;
            if (!scenarioMetrics) {
                console.warn('No gridded metrics for scenario', {
                    datasetKey,
                    scenarioKey,
                    model,
                    requestedKey: scenarioDatasetKey,
                    availableKeys: Object.keys(metricsScenarios).filter((key) =>
                        key.startsWith(`${datasetKey}_${scenarioKey}_`)
                    ),
                });
                return [];
            }
            if (resolvedKey !== scenarioDatasetKey) {
                console.log(
                    `Resolved metrics scenario key: ${scenarioDatasetKey} -> ${resolvedKey}`
                );
            }

            const resolution = Number(
                metricsData.grid_resolution_deg
                    ?? gridData.grid_resolution_deg
                    ?? griddedEntry?.grid_resolution_deg
                    ?? siteGridResolutionDeg
            );
            if (!Number.isFinite(resolution) || resolution <= 0) {
                console.warn('Missing grid_resolution_deg for', datasetKey);
                return [];
            }
            const getMetric = (name, idx) => {
                const arr = scenarioMetrics[name];
                if (!Array.isArray(arr)) return 0;
                const value = Number(arr[idx] ?? 0);
                return Number.isFinite(value) ? value : 0;
            };

            // ── Polygon path (e.g. tourism reef polygons) ──────────────────────
            if (isPolygonGeojson) {
                return gridData.features
                    .map((feature) => {
                        const ci = feature.id ?? feature.properties?.i;
                        if (ci == null) return null;
                        const valueLoss = getMetric('value_loss', ci);
                        const lossFraction = getMetric('loss_fraction', ci);
                        const originalValue = Number(feature.properties?.ov ?? 0);
                        if (originalValue <= 0 && valueLoss <= 0 && lossFraction <= 0) {
                            return null;
                        }
                        return {
                            ...feature,
                            properties: {
                                site_id: ci,
                                country: feature.properties?.co || '',
                                value_type: datasetKey,
                                original_value: originalValue,
                                n_sites: Number(feature.properties?.n ?? 1),
                                grid_resolution_deg: resolution,
                                value_loss: valueLoss,
                                loss_fraction: lossFraction,
                                coral_change: getMetric('coral_change', ci),
                                annual_loss: getMetric('annual_loss', ci),
                                cumulative_loss: getMetric('cumulative_loss', ci),
                                cumulative_loss_fraction: getMetric('cumulative_loss_fraction', ci),
                            },
                        };
                    })
                    .filter(Boolean);
            }

            // ── Grid-cell path (fisheries, coastal protection) ─────────────────
            return gridData.cells
                .map((cell, idx) => {
                    const valueLoss = getMetric('value_loss', idx);
                    const lossFraction = getMetric('loss_fraction', idx);
                    const originalValue = Number(cell.ov ?? 0);
                    if (originalValue <= 0 && valueLoss <= 0 && lossFraction <= 0) {
                        return null;
                    }
                    return {
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [Number(cell.lon), Number(cell.lat)],
                        },
                        properties: {
                            site_id: cell.i,
                            country: cell.co || '',
                            value_type: datasetKey,
                            original_value: originalValue,
                            n_sites: Number(cell.n ?? 1),
                            grid_resolution_deg: resolution,
                            value_loss: valueLoss,
                            loss_fraction: lossFraction,
                            coral_change: getMetric('coral_change', idx),
                            annual_loss: getMetric('annual_loss', idx),
                            cumulative_loss: getMetric('cumulative_loss', idx),
                            cumulative_loss_fraction: getMetric('cumulative_loss_fraction', idx),
                        },
                    };
                })
                .filter(Boolean);
        }

        async function loadDatasetWidePointTileFeatures({
            datasetKey,
            scenarioDatasetKey,
            tileKeys,
            datasetTileIndex,
        }) {
            const datasetEntry = datasetTileIndex?.[datasetKey];
            if (!datasetEntry) {
                return [];
            }
            const geometryFiles = datasetEntry.geometry || {};
            const attributeFiles = datasetEntry.attributes || {};
            const keysToLoad = tileKeys && tileKeys.length > 0
                ? tileKeys
                : Object.keys(geometryFiles);

            const perTileFeatures = await Promise.all(
                keysToLoad.map(async (tileKey) => {
                    const geomFile = geometryFiles[tileKey];
                    const attrFile = attributeFiles[tileKey];
                    if (!geomFile || !attrFile) {
                        return [];
                    }
                    const [geomData, attrData] = await Promise.all([
                        fetchGeoJsonWithCache(geomFile),
                        fetchGeoJsonWithCache(attrFile),
                    ]);
                    if (!geomData || !Array.isArray(geomData.features)) {
                        return [];
                    }
                    const scenarioColumns = attrData?.scenario_metrics?.[scenarioDatasetKey] || {};
                    const getMetric = (name, idx) => {
                        const arr = scenarioColumns?.[name];
                        if (!Array.isArray(arr)) return 0;
                        const value = Number(arr[idx] || 0);
                        return Number.isFinite(value) ? value : 0;
                    };
                    return geomData.features.map((feature, idx) => ({
                        type: 'Feature',
                        geometry: feature.geometry,
                        properties: {
                            ...(feature.properties || {}),
                            value_loss: getMetric('value_loss', idx),
                            loss_fraction: getMetric('loss_fraction', idx),
                            coral_change: getMetric('coral_change', idx),
                            annual_loss: getMetric('annual_loss', idx),
                            cumulative_loss: getMetric('cumulative_loss', idx),
                            cumulative_loss_fraction: getMetric('cumulative_loss_fraction', idx),
                        },
                    }));
                })
            );

            return perTileFeatures.flat();
        }

        /**
         * Load map features per dataset, mixing gridded layers (tourism, fisheries)
         * with viewport-tiled point layers (coastal protection).
         */
        async function loadSelectedSiteFeatures({
            datasetsToLoad,
            scenarioKey,
            model,
            isCumulative,
            buildScenarioKey,
            griddedManifest,
            datasetTileIndex,
            datasetTileZoom,
            mapBounds,
            mapZoom,
        }) {
            const minPointMapZoom = Number(siteManifest?.site_dataset_min_map_zoom ?? 4);
            const canLoadPointsAtZoom = mapZoom >= minPointMapZoom;
            const effectiveTileZoom = Number.isFinite(datasetTileZoom) ? datasetTileZoom : null;
            const visibleTileKeys = effectiveTileZoom && mapBounds
                ? getTileKeysForBounds(mapBounds, effectiveTileZoom)
                : [];

            const featureLists = await Promise.all(
                datasetsToLoad.map(async (datasetKey) => {
                    const tileEntry = datasetTileIndex?.[datasetKey];
                    if (tileEntry?.geometry && tileEntry?.attributes) {
                        if (!canLoadPointsAtZoom) {
                            return [];
                        }
                        return loadDatasetWidePointTileFeatures({
                            datasetKey,
                            scenarioDatasetKey: buildScenarioKey(datasetKey),
                            tileKeys: visibleTileKeys,
                            datasetTileIndex,
                        });
                    }

                    const griddedEntry = griddedManifest?.[datasetKey];
                    if (griddedEntry) {
                        return loadGriddedSiteFeatures({
                            datasetKey,
                            scenarioKey,
                            model,
                            scenarioDatasetKey: buildScenarioKey(datasetKey),
                            griddedEntry,
                            isCumulative,
                        });
                    }

                    console.warn('No map data source for dataset', datasetKey);
                    return [];
                })
            );

            const skippedPointDatasets = datasetsToLoad.filter((datasetKey) => {
                const tileEntry = datasetTileIndex?.[datasetKey];
                return tileEntry?.geometry && tileEntry?.attributes && !canLoadPointsAtZoom;
            });

            return {
                features: featureLists.flat(),
                skippedPointDatasets,
                minPointMapZoom,
                canLoadPointsAtZoom,
            };
        }

        function getSelectedValueType(controlId, fallback = 'all') {
            const control = document.getElementById(controlId);
            return control ? control.value : fallback;
        }

        function formatValueType(valueType) {
            return VALUE_TYPE_LABELS[valueType] || valueType;
        }

        function describeValueType(valueType) {
            return VALUE_TYPE_DESCRIPTIONS[valueType] || 'Reef-associated economic value layer.';
        }

        function getScenarioComparisonValueTypes() {
            const checkboxes = document.querySelectorAll('.scenario-value-type-checkbox');
            const selected = Array.from(checkboxes)
                .filter((cb) => cb.checked)
                .map((cb) => cb.value);
            return selected;
        }

        function getOverviewSelectedValueTypes() {
            const checkboxes = document.querySelectorAll('.overview-value-type-checkbox');
            return Array.from(checkboxes)
                .filter((cb) => cb.checked)
                .map((cb) => cb.value);
        }

        function valueTypeColor(valueType) {
            return CHART_COLORS.dataset?.[valueType] || '#94a3b8';
        }

        function scenarioColor(scenarioKey) {
            const key = String(scenarioKey || '').toLowerCase();
            if (key.includes('rcp85')) return CHART_COLORS.scenario?.rcp85 || '#e74c3c';
            if (key.includes('rcp45')) return CHART_COLORS.scenario?.rcp45 || '#3498db';
            return '#94a3b8';
        }

        function getMapSelectedValueTypes() {
            const checkboxes = document.querySelectorAll('.map-value-type-checkbox');
            return Array.from(checkboxes)
                .filter((cb) => cb.checked)
                .map((cb) => cb.value);
        }

        function getTrajectorySelectedValueTypes() {
            const checkboxes = document.querySelectorAll('.traj-value-type-checkbox');
            return Array.from(checkboxes)
                .filter((cb) => cb.checked)
                .map((cb) => cb.value);
        }

        function formatBillions(valueInBillions) {
            const abs = Math.abs(Number(valueInBillions || 0));
            if (abs >= 1) return `$${valueInBillions.toFixed(2)}B`;
            return `$${(valueInBillions * 1000).toFixed(1)}M`;
        }

        function aggregateRowsByCountry(rows, isCumulative = false, includeGdp = false) {
            const grouped = new Map();
            rows.forEach((row) => {
                const country = row.country || '';
                const key = `${country}||${row.iso_a3 || ''}`;
                if (!grouped.has(key)) {
                    grouped.set(key, {
                        country,
                        iso_a3: row.iso_a3 || '',
                        original_value: 0,
                        value_loss: 0,
                        cumulative_loss: 0,
                        annual_loss: 0,
                        loss_fraction: 0,
                        cumulative_loss_fraction: 0,
                        _annual_fraction_weighted_loss: 0,
                        _cumulative_fraction_weighted_loss: 0,
                        _gdp_loss: 0,
                        _gdp_base: null,
                    });
                }
                const acc = grouped.get(key);
                acc.original_value += Number(row.original_value || 0);
                acc.value_loss += Number(row.value_loss || 0);
                acc.cumulative_loss += Number(row.cumulative_loss || 0);
                acc.annual_loss += Number(row.annual_loss || row.value_loss || 0);
                acc._annual_fraction_weighted_loss += Number(row.original_value || 0) * Number(row.loss_fraction || 0);
                acc._cumulative_fraction_weighted_loss += Number(row.original_value || 0) * Number(row.cumulative_loss_fraction || 0);

                if (includeGdp) {
                    const gdpKey = `${row.scenario}||${row.model}||${row.value_type || ''}||${country}`;
                    const gdpRec = window.gdpImpactLookup ? window.gdpImpactLookup[gdpKey] : null;
                    if (gdpRec) {
                        acc._gdp_loss += Number(gdpRec.value_loss || 0);
                        const nationalGdp = Number(gdpRec.national_gdp || 0);
                        if (nationalGdp > 0) {
                            acc._gdp_base = nationalGdp;
                        }
                    }
                }
            });

            return Array.from(grouped.values()).map((row) => {
                const original = row.original_value;
                row.loss_fraction = original > 0
                    ? (row._annual_fraction_weighted_loss > 0
                        ? row._annual_fraction_weighted_loss / original
                        : row.value_loss / original)
                    : 0;
                row.cumulative_loss_fraction = original > 0
                    ? (row._cumulative_fraction_weighted_loss > 0
                        ? row._cumulative_fraction_weighted_loss / original
                        : row.cumulative_loss / original)
                    : 0;
                if (includeGdp) {
                    row.loss_as_gdp_pct = row._gdp_base && row._gdp_base > 0
                        ? (row._gdp_loss / row._gdp_base) * 100
                        : 0;
                }
                delete row._annual_fraction_weighted_loss;
                delete row._cumulative_fraction_weighted_loss;
                delete row._gdp_loss;
                delete row._gdp_base;
                return row;
            });
        }
        
        function sanitizeModelKey(name) {
            return String(name)
                .replace(/\s+/g, '_')
                .replace(/\//g, '_')
                .replace(/%/g, 'pct')
                .replace(/[()]/g, '');
        }

        function shortModelLabel(name) {
            if (String(name).includes('Linear')) return 'Linear';
            if (String(name).includes('Compound')) return 'Compound';
            if (String(name).includes('Tipping')) return 'Tipping Point';
            return String(name);
        }

        function isLinearModelName(name) {
            const label = shortModelLabel(name);
            return label === 'Linear';
        }

        function modelColor(name) {
            const label = shortModelLabel(name);
            if (label === 'Tipping Point') return CHART_COLORS.model?.tipping || '#dc2626';
            if (label === 'Compound') return CHART_COLORS.model?.compound || '#F11B00';
            if (label === 'Linear') return CHART_COLORS.model?.linear || '#E3B710';
            return '#94a3b8';
        }

        function sortModels(models) {
            return [...models].sort((a, b) => {
                const ai = MODEL_SORT_ORDER.indexOf(shortModelLabel(a));
                const bi = MODEL_SORT_ORDER.indexOf(shortModelLabel(b));
                return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
            });
        }

        function sortScenarios(scenarios) {
            const rank = (scenario) => {
                const s = String(scenario).toLowerCase();
                const rcp = s.includes('rcp85') ? 2 : 1;
                const year = s.includes('2100') ? 2 : 1;
                return rcp * 10 + year;
            };
            return [...scenarios].sort((a, b) => rank(a) - rank(b));
        }

        function lossColorscale(stops) {
            const [low, mid, high, extreme] = CHART_COLORS.lossScale || [];
            return [
                [0, low || '#22c55e'],
                [stops[0], mid || '#eab308'],
                [stops[1], high || '#E3B710'],
                [1, extreme || '#F11B00'],
            ];
        }

        function getHabitatExportA() {
            return CHART_COLORS.habitatExportA ?? 0.4;
        }

        function getIllustrationHabitatA() {
            const slider = document.getElementById('habitat-illustration-slider');
            if (!slider || slider.value === '') {
                return getHabitatExportA();
            }
            const idx = Number(slider.value);
            return HABITAT_ILLUSTRATION_OPTIONS[idx] ?? getHabitatExportA();
        }

        function syncIllustrationHabitatControl() {
            const slider = document.getElementById('habitat-illustration-slider');
            const output = document.querySelector('.habitat-illustration-value');
            const a = getIllustrationHabitatA();
            if (output) {
                output.textContent = String(a);
            }
            if (slider) {
                slider.setAttribute('aria-valuenow', String(slider.value));
                slider.setAttribute('aria-valuetext', String(a));
            }
        }

        function updateMapHabitatBadge() {
            const wrap = document.getElementById('map-habitat-alpha-wrap');
            const badge = document.getElementById('map-habitat-alpha');
            const model = document.getElementById('map-model')?.value || '';
            if (!wrap || !badge) return;
            const show = String(model).includes('Linear');
            wrap.classList.toggle('is-hidden', !show);
            badge.textContent = `Fisheries habitat α = ${getHabitatExportA()}`;
        }

        function scaleFisheriesLinearLoss(loss, habitatA) {
            const lossValue = Number(loss || 0);
            const aRef = getHabitatExportA();
            if (!lossValue || habitatA === aRef) return lossValue;
            const denom = 1 - aRef;
            if (denom <= 0) return lossValue;
            return lossValue * (1 - habitatA) / denom;
        }

        function mapModelToCountryModel(mapModelValue) {
            if (!mapModelValue) return mapModelValue;
            if (String(mapModelValue).includes('(')) return mapModelValue;
            const models = summaryData?.snapshot_results
                ? [...new Set(summaryData.snapshot_results.map((r) => r.model))]
                : countryData
                  ? [...new Set(countryData.map((c) => c.model))]
                  : [];
            const target = sanitizeModelKey(mapModelValue).toLowerCase();
            return (
                models.find((m) => sanitizeModelKey(m).toLowerCase() === target) ||
                mapModelValue
            );
        }

        function resolveScenarioDatasetKey(datasetKey, scenarioKey, model, isCumulative) {
            const manifestKeys = isCumulative
                ? siteManifest?.cumulative_scenarios || []
                : siteManifest?.scenarios || [];
            const prefix = `${datasetKey}_${scenarioKey}_`;

            const suffixCandidates = [];
            const addSuffix = (suffix) => {
                if (suffix && !suffixCandidates.includes(suffix)) {
                    suffixCandidates.push(suffix);
                }
            };

            addSuffix(sanitizeModelKey(model));
            if (!String(model).includes('(')) {
                addSuffix(model);
            }
            const countryModel = mapModelToCountryModel(model);
            if (countryModel && countryModel !== model) {
                addSuffix(sanitizeModelKey(countryModel));
            }
            if (String(model).toLowerCase().includes('linear')) {
                addSuffix('Linear_3.81pct_relpct');
                addSuffix('Linear_3.81pct_pp');
                addSuffix(sanitizeModelKey('Linear (3.81%/rel%)'));
                addSuffix(sanitizeModelKey('Linear (3.81%/pp)'));
            }

            for (const suffix of suffixCandidates) {
                const key = `${prefix}${suffix}`;
                if (manifestKeys.includes(key)) {
                    return key;
                }
            }

            const modelLower = String(model).toLowerCase();
            const fallback = manifestKeys.find((key) => {
                if (!key.startsWith(prefix)) return false;
                if (modelLower.includes('linear')) return key.includes('Linear');
                if (modelLower.includes('compound')) return key.includes('Compound');
                if (modelLower.includes('tipping')) return key.includes('Tipping');
                return false;
            });
            return fallback || `${prefix}${sanitizeModelKey(countryModel || model)}`;
        }

        /** Resolve scenario key against actual metrics JSON keys (not cached manifest). */
        function resolveMetricsScenarioKey(metricsScenarios, datasetKey, scenarioKey, model) {
            const keys = Object.keys(metricsScenarios || {});
            if (!keys.length) return null;

            const prefix = `${datasetKey}_${scenarioKey}_`;
            const countryModel = mapModelToCountryModel(model) || model;
            const suffixCandidates = [];
            const addSuffix = (suffix) => {
                if (suffix && !suffixCandidates.includes(suffix)) {
                    suffixCandidates.push(suffix);
                }
            };

            addSuffix(sanitizeModelKey(model));
            if (!String(model).includes('(')) {
                addSuffix(model);
            }
            if (countryModel && countryModel !== model) {
                addSuffix(sanitizeModelKey(countryModel));
            }
            if (String(model).toLowerCase().includes('linear') ||
                String(countryModel).toLowerCase().includes('linear')) {
                addSuffix('Linear_3.81pct_relpct');
                addSuffix('Linear_3.81pct_pp');
                addSuffix(sanitizeModelKey('Linear (3.81%/rel%)'));
                addSuffix(sanitizeModelKey('Linear (3.81%/pp)'));
            }
            if (String(model).toLowerCase().includes('compound') ||
                String(countryModel).toLowerCase().includes('compound')) {
                addSuffix('Compound_3.81pct_pp');
                addSuffix(sanitizeModelKey('Compound (3.81%/pp)'));
            }
            if (String(model).toLowerCase().includes('tipping') ||
                String(countryModel).toLowerCase().includes('tipping')) {
                addSuffix('Tipping_Point_threshold=10pct');
                addSuffix(sanitizeModelKey('Tipping Point (threshold=10%)'));
            }

            for (const suffix of suffixCandidates) {
                const key = `${prefix}${suffix}`;
                if (metricsScenarios[key]) {
                    return key;
                }
            }

            const modelLower = String(countryModel).toLowerCase();
            return keys.find((key) => {
                if (!key.startsWith(prefix)) return false;
                if (modelLower.includes('linear')) return key.includes('Linear');
                if (modelLower.includes('compound')) return key.includes('Compound');
                if (modelLower.includes('tipping')) return key.includes('Tipping');
                return false;
            }) || null;
        }

        function populateModelSelectors() {
            const models = summaryData?.snapshot_results
                ? [...new Set(summaryData.snapshot_results.map((r) => r.model))]
                : countryData
                  ? [...new Set(countryData.map((c) => c.model))]
                  : [];
            if (!models.length) return;

            const MAP_MODEL_LEGACY_VALUES = {
                'Linear_3.81pct_relpct': 'Linear (3.81%/rel%)',
                'Linear_3.81pct_pp': 'Linear (3.81%/rel%)',
                'Linear (3.81%/pp)': 'Linear (3.81%/rel%)',
                'Linear (3.81%/rel%)': 'Linear (3.81%/rel%)',
                'Compound_3.81pct_pp': 'Compound (3.81%/pp)',
                'Compound (3.81%/pp)': 'Compound (3.81%/pp)',
                'Tipping_Point_threshold=10pct': 'Tipping Point (threshold=10%)',
                'Tipping Point (threshold=10%)': 'Tipping Point (threshold=10%)',
            };

            const fillSelect = (selectId, { useSanitized = false } = {}) => {
                const select = document.getElementById(selectId);
                if (!select) return;
                const current = MAP_MODEL_LEGACY_VALUES[select.value] || select.value;
                select.innerHTML = '';
                models.forEach((model) => {
                    const option = document.createElement('option');
                    option.value = useSanitized ? sanitizeModelKey(model) : model;
                    option.textContent = shortModelLabel(model);
                    select.appendChild(option);
                });
                const values = [...select.options].map((o) => o.value);
                if (values.includes(current)) {
                    select.value = current;
                } else if (values.length) {
                    select.value = values[0];
                }
            };

            fillSelect('country-model');
            fillSelect('gdp-comparison-model');
            fillSelect('gdp-model');
            fillSelect('map-model');
        }

        function pathnameParts() {
            const parts = window.location.pathname.replace(/\/$/, '').split('/').filter(Boolean);
            if (parts.length && parts[parts.length - 1] === 'index.html') {
                parts.pop();
            }
            return parts;
        }

        /** Derive app root from dashboard.js location (works for /docs/, GitHub Pages, and /map routes). */
        function detectAppBaseFromScript() {
            const scripts = document.querySelectorAll('script[src*="dashboard.js"]');
            const scriptEl = scripts[scripts.length - 1];
            if (!scriptEl?.src) {
                return null;
            }
            try {
                const url = new URL(scriptEl.src, window.location.href);
                return url.pathname.replace(/\/assets\/js\/dashboard\.js(?:\?.*)?$/, '') || '';
            } catch {
                return null;
            }
        }

        function initAppBasePath() {
            const scriptBase = detectAppBaseFromScript();
            if (scriptBase !== null) {
                APP_BASE_PATH = scriptBase;
            } else {
                const parts = pathnameParts();
                const last = parts[parts.length - 1] || '';

                if (PAGE_IDS.includes(last)) {
                    APP_BASE_PATH = parts.length > 1 ? `/${parts.slice(0, -1).join('/')}` : '';
                } else if (parts.length >= 1) {
                    APP_BASE_PATH = `/${parts.join('/')}`;
                } else {
                    APP_BASE_PATH = '';
                }
            }

            // Always use absolute paths so fetches work regardless of current /map vs /docs/map URL.
            DATA_PATH = APP_BASE_PATH
                ? `${APP_BASE_PATH}/exported_data/`
                : '/exported_data/';
            siteFileCache.clear();
            console.log('Dashboard data path:', DATA_PATH);
        }

        async function fetchJsonData(filename) {
            const url = `${DATA_PATH}${filename}?v=${DATA_VERSION}`;
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to load ${url} (${response.status})`);
            }
            return response.json();
        }

        function getHashRoute() {
            return window.location.hash.slice(1) || '';
        }

        function getMethodsSectionFromHash() {
            const hash = getHashRoute();
            return hash.startsWith('methods-') ? hash : null;
        }

        function getPageFromPath() {
            const hash = getHashRoute();
            if (hash) {
                if (hash === 'models') {
                    return 'methods';
                }
                if (PAGE_IDS.includes(hash)) {
                    return hash;
                }
                if (hash.startsWith('methods-')) {
                    return 'methods';
                }
            }

            // Pathname fallback (GitHub Pages 404.html / serve_docs.py SPA mode).
            const parts = pathnameParts();
            const last = parts[parts.length - 1] || '';
            if (last === 'models') {
                return 'methods';
            }
            return PAGE_IDS.includes(last) ? last : DEFAULT_PAGE;
        }

        /** Rewrite /docs/methods-style paths to #methods so refresh works on static servers. */
        function migratePathUrlToHash() {
            const hash = getHashRoute();
            if (hash) {
                return;
            }
            const parts = pathnameParts();
            const last = parts[parts.length - 1] || '';
            if (!last || last === 'index.html') {
                return;
            }
            let page = null;
            if (last === 'models') {
                page = 'methods';
            } else if (PAGE_IDS.includes(last)) {
                page = last;
            }
            if (!page) {
                return;
            }
            const base = APP_BASE_PATH ? `${APP_BASE_PATH}/` : '/';
            history.replaceState({ page }, '', `${base}#${page}`);
        }

        function buildPageUrl(page, { sectionId = null } = {}) {
            const targetPage = PAGE_IDS.includes(page) ? page : DEFAULT_PAGE;
            const base = APP_BASE_PATH ? `${APP_BASE_PATH}/` : '/';
            if (sectionId) {
                return `${base}#${sectionId}`;
            }
            if (targetPage === DEFAULT_PAGE) {
                return base;
            }
            return `${base}#${targetPage}`;
        }

        function updateDocumentTitle(page) {
            const section = PAGE_TITLES[page] || PAGE_TITLES[DEFAULT_PAGE];
            document.title = `${section} | Coral Reef Economics`;
        }

        function updateNavLinks() {
            document.querySelectorAll('.nav-link[data-page]').forEach((link) => {
                link.href = buildPageUrl(link.dataset.page);
            });
            const brand = document.getElementById('nav-brand-home');
            if (brand) {
                brand.href = buildPageUrl(DEFAULT_PAGE);
            }
        }

        function activatePage(page) {
            const targetPage = PAGE_IDS.includes(page) ? page : DEFAULT_PAGE;

            document.querySelectorAll('.nav-link[data-page]').forEach((link) => {
                link.classList.toggle('active', link.dataset.page === targetPage);
            });

            document.querySelectorAll('.page').forEach((section) => {
                section.classList.remove('active');
            });
            const pageEl = document.getElementById(`page-${targetPage}`);
            if (pageEl) {
                pageEl.classList.add('active');
            }

            updateDocumentTitle(targetPage);

            if (targetPage === 'map') {
                setTimeout(() => {
                    if (map) {
                        map.invalidateSize();
                    }
                    loadSiteData();
                }, 100);
            } else if (targetPage === 'trajectories') {
                renderTrajectoryPage();
            } else if (targetPage === 'gdp') {
                renderCountryChart();
                renderGdpComparison();
            } else if (targetPage === 'methods') {
                renderModelComparison();
                renderMethodsEnvVisualizations();
                initMethodsPage();
                setTimeout(updateMethodsTocActive, 200);
            }
        }

        function getMethodsScrollMargin() {
            const navInner = document.querySelector('.nav-inner');
            return (navInner?.offsetHeight || 64) + 16;
        }

        function updateMethodsTocActive() {
            const page = document.getElementById('page-methods');
            if (!page?.classList.contains('active')) return;

            const sections = document.querySelectorAll('.methods-sections .method-box[id]');
            const links = document.querySelectorAll('.methods-toc a[href^="#"]');
            if (!sections.length || !links.length) return;

            const margin = getMethodsScrollMargin();
            let currentId = sections[0]?.id || null;
            sections.forEach((section) => {
                if (section.getBoundingClientRect().top <= margin + 12) {
                    currentId = section.id;
                }
            });

            links.forEach((link) => {
                const id = link.getAttribute('href')?.slice(1);
                link.classList.toggle('is-active', id === currentId);
            });
        }

        function bindMethodsTocScrollSpy() {
            if (window._methodsTocScrollSpyBound) return;
            window._methodsTocScrollSpyBound = true;
            window.addEventListener('scroll', updateMethodsTocActive, { passive: true });
            window.addEventListener('resize', updateMethodsTocActive);
        }

        function updateMethodsScrollMargins() {
            const margin = `${getMethodsScrollMargin()}px`;
            document.querySelectorAll('.methods-sections .method-box[id]').forEach((box) => {
                box.style.scrollMarginTop = margin;
            });
        }

        function scrollToMethodsSection(sectionId) {
            const el = document.getElementById(sectionId);
            if (!el) return;
            updateMethodsScrollMargins();
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function initMethodsPage() {
            const toc = document.querySelector('.methods-toc');
            updateMethodsScrollMargins();
            bindMethodsTocScrollSpy();
            updateMethodsTocActive();
            if (!window._methodsScrollMarginBound) {
                window._methodsScrollMarginBound = true;
                window.addEventListener('resize', updateMethodsScrollMargins);
            }
            if (toc && toc.dataset.bound !== '1') {
                toc.dataset.bound = '1';
                toc.querySelectorAll('a[href^="#"]').forEach((link) => {
                    link.addEventListener('click', (event) => {
                        const id = link.getAttribute('href')?.slice(1);
                        if (!id) return;
                        event.preventDefault();
                        scrollToMethodsSection(id);
                        const url = buildPageUrl('methods', { sectionId: id });
                        history.replaceState({ page: 'methods', section: id }, '', url);
                        if (window.matchMedia('(max-width: 1023px)').matches) {
                            link.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
                        }
                    });
                });
            }
            const sectionId = getMethodsSectionFromHash();
            if (sectionId) {
                setTimeout(() => scrollToMethodsSection(sectionId), 150);
            }
        }

        function navigateToPage(page, { replace = false } = {}) {
            const targetPage =
                page === 'models' || page === 'methods'
                    ? 'methods'
                    : PAGE_IDS.includes(page)
                        ? page
                        : DEFAULT_PAGE;

            const url = buildPageUrl(targetPage);
            activatePage(targetPage);

            if (replace) {
                history.replaceState({ page: targetPage }, '', url);
            } else {
                history.pushState({ page: targetPage }, '', url);
            }
        }

        async function loadData() {
            initAppBasePath();
            migratePathUrlToHash();
            try {
                const [summary, trajectories, countries, cumulativeCountries, curves, manifest, gdpImpacts, envLocations, envQdm, envCyclone, envReefcheck, cotwEcoregions] = await Promise.all([
                    fetchJsonData('summary.json'),
                    fetchJsonData('trajectories.json'),
                    fetchJsonData('country_results.json'),
                    fetchJsonData('cumulative_country_results.json').catch(() => []),
                    fetchJsonData('model_curves.json'),
                    fetchJsonData('manifest.json'),
                    fetchJsonData('gdp_impacts.json').catch(() => null),
                    fetchJsonData('methods_env_locations.json').catch(() => null),
                    fetchJsonData('methods_env_qdm.json').catch(() => null),
                    fetchJsonData('methods_env_cyclone_grid.json').catch(() => null),
                    fetchJsonData('methods_env_reefcheck.json').catch(() => null),
                    fetchJsonData('methods_cotw_ecoregions.json').catch(() => null),
                ]);
                
                summaryData = summary;
                trajectoryData = trajectories;
                countryData = countries;
                cumulativeCountryData = cumulativeCountries;
                gdpImpactData = gdpImpacts;
                // Build lookup for GDP impact by (scenario, model, country)
                window.gdpImpactLookup = {};
                if (Array.isArray(gdpImpactData)) {
                    gdpImpactData.forEach(d => {
                        const key = `${d.scenario}||${d.model}||${d.value_type || ''}||${d.country}`;
                        window.gdpImpactLookup[key] = d;
                    });
                }
                modelCurves = curves;
                methodsEnvData = (envLocations || envQdm || envCyclone || envReefcheck || cotwEcoregions) ? {
                    locations: envLocations,
                    qdm: envQdm,
                    cyclone: envCyclone,
                    reefcheck: envReefcheck,
                    cotwEcoregions: cotwEcoregions,
                } : null;
                siteManifest = manifest;
                const manifestRes = Number(manifest?.cell_resolution_deg);
                if (Number.isFinite(manifestRes) && manifestRes > 0) {
                    siteGridResolutionDeg = manifestRes;
                } else {
                    const annual = manifest?.gridded_sites_annual || {};
                    for (const entry of Object.values(annual)) {
                        const res = Number(entry?.grid_resolution_deg);
                        if (Number.isFinite(res) && res > 0) {
                            siteGridResolutionDeg = res;
                            break;
                        }
                    }
                }
                
                console.log('Data loaded successfully:', {
                    summary: !!summary,
                    trajectories: trajectories?.length || 0,
                    countries: countries?.length || 0,
                    cumulativeCountries: cumulativeCountries?.length || 0,
                    gdpImpacts: !!gdpImpacts,
                    curves: !!curves,
                    manifest: !!manifest
                });
                
                if (cumulativeCountries && cumulativeCountries.length > 0) {
                    const uniqueScenarios = [...new Set(cumulativeCountries.map(c => c.scenario))];
                    const uniqueModels = [...new Set(cumulativeCountries.map(c => c.model))];
                    console.log('Cumulative data loaded:', {
                        totalRecords: cumulativeCountries.length,
                        uniqueScenarios,
                        uniqueModels
                    });
                } else {
                    console.warn('No cumulative country data loaded');
                }
                
                initializeDashboard();
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('summary-stats').innerHTML =
                    `<p style="color: var(--accent-red);">Error loading data from <code>${DATA_PATH}</code>. ` +
                    'Serve the site from the <code>docs/</code> folder or open <code>/docs/</code> if using the repo root. ' +
                    `Details: ${error.message}</p>`;
            }
        }
        
        // ============================================================
        // INITIALIZATION
        // ============================================================
        
        function initializeDashboard() {
            initChartColors();
            populateModelSelectors();
            renderSummaryStats();
            renderScenarioComparison();
            renderOverviewTrajectory('cumulative_loss');
            renderModelComparison();
            initializeMap();
            updateMapHabitatBadge();

            setupNavigation();
            updateNavLinks();
            navigateToPage(getPageFromPath(), { replace: true });
            setupControls();
        }

        function setupNavigation() {
            const menuToggle = document.getElementById('nav-menu-toggle');
            const mobileMenu = document.getElementById('nav-mobile-menu');

            if (menuToggle && mobileMenu) {
                menuToggle.addEventListener('click', () => {
                    menuToggle.classList.toggle('active');
                    mobileMenu.classList.toggle('active');
                });

                document.addEventListener('click', (e) => {
                    if (!menuToggle.contains(e.target) && !mobileMenu.contains(e.target)) {
                        menuToggle.classList.remove('active');
                        mobileMenu.classList.remove('active');
                    }
                });
            }

            document.querySelectorAll('.nav-link[data-page]').forEach((link) => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const page = link.dataset.page;
                    if (!page) {
                        return;
                    }

                    if (menuToggle && mobileMenu) {
                        menuToggle.classList.remove('active');
                        mobileMenu.classList.remove('active');
                    }

                    navigateToPage(page);
                });
            });

            const brand = document.getElementById('nav-brand-home');
            if (brand) {
                brand.addEventListener('click', (e) => {
                    e.preventDefault();
                    navigateToPage(DEFAULT_PAGE);
                });
            }

            window.addEventListener('popstate', (event) => {
                const page = event.state?.page || getPageFromPath();
                activatePage(page);
            });

            window.addEventListener('hashchange', () => {
                const page = getPageFromPath();
                activatePage(page);
            });
        }
        
        function setupControls() {
            // Overview trajectory toggle
            document.querySelectorAll('.toggle-btn[data-metric]').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    document.querySelectorAll('.toggle-btn[data-metric]').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    const currentMetric = e.target.dataset.metric;
                    renderOverviewTrajectory(currentMetric);
                });
            });
            
            // Overview model selector
            document.getElementById('overview-model').addEventListener('change', () => {
                const activeBtn = document.querySelector('.toggle-btn[data-metric].active');
                const metric = activeBtn ? activeBtn.dataset.metric : 'cumulative_loss';
                renderOverviewTrajectory(metric);
            });
            document.querySelectorAll('.overview-value-type-checkbox').forEach((checkbox) => {
                checkbox.addEventListener('change', () => {
                    const activeBtn = document.querySelector('.toggle-btn[data-metric].active');
                    const metric = activeBtn ? activeBtn.dataset.metric : 'cumulative_loss';
                    renderOverviewTrajectory(metric);
                });
            });
            document.querySelectorAll('.scenario-value-type-checkbox').forEach((checkbox) => {
                checkbox.addEventListener('change', () => {
                    renderScenarioComparison();
                });
            });
            
            // Map controls
            // Debounce filter changes to avoid excessive re-renders
            const debouncedLoadSiteData = () => {
                clearTimeout(renderTimeout);
                renderTimeout = setTimeout(() => {
                    loadSiteData();
                }, 200);  // 200ms debounce
            };
            
            document.getElementById('map-scenario').addEventListener('change', debouncedLoadSiteData);
            document.getElementById('map-model').addEventListener('change', () => {
                updateMapHabitatBadge();
                debouncedLoadSiteData();
            });
            document.querySelectorAll('.map-value-type-checkbox').forEach((checkbox) => {
                checkbox.addEventListener('change', debouncedLoadSiteData);
            });
            document.getElementById('map-metric').addEventListener('change', () => {
                // Debounce metric changes
                clearTimeout(renderTimeout);
                renderTimeout = setTimeout(() => {
                    // Re-render sites and choropleth with new metric
                    if (currentSiteGeojson) {
                        // Preserve current map view when changing metric
                        let currentView = null;
                        if (map) {
                            currentView = {
                                center: map.getCenter(),
                                zoom: map.getZoom()
                            };
                        }
                        const scenario = document.getElementById('map-scenario').value;
                        const isCumulative = scenario.startsWith('cumulative_');
                        renderSites(currentSiteGeojson, isCumulative, currentView);
                    }
                    if (document.getElementById('map-choropleth-toggle').checked) {
                        renderChoropleth();
                    }
                }, 150);
            });
            document.getElementById('map-choropleth-toggle').addEventListener('change', toggleChoropleth);
            
            // Trajectory controls
            document.getElementById('traj-interpolation').addEventListener('change', renderTrajectoryPage);
            document.getElementById('traj-model').addEventListener('change', renderTrajectoryPage);
            document.querySelectorAll('.traj-value-type-checkbox').forEach((checkbox) => {
                checkbox.addEventListener('change', renderTrajectoryPage);
            });
            
            // Country controls
            document.getElementById('country-scenario').addEventListener('change', renderCountryChart);
            document.getElementById('country-model').addEventListener('change', renderCountryChart);
            document.getElementById('country-value-type').addEventListener('change', renderCountryChart);
            document.getElementById('country-limit').addEventListener('change', renderCountryChart);
            document.getElementById('country-metric').addEventListener('change', renderCountryChart);
            document.getElementById('country-color-mode').addEventListener('change', renderCountryChart);
            
            // GDP comparison controls
            document.getElementById('gdp-comparison-model').addEventListener('change', renderGdpComparison);
            document.getElementById('gdp-value-type').addEventListener('change', renderGdpComparison);
            document.getElementById('gdp-comparison-metric').addEventListener('change', renderGdpComparison);
            document.getElementById('gdp-comparison-limit').addEventListener('change', renderGdpComparison);

            const modelCoverSlider = document.getElementById('model-initial-cover');
            if (modelCoverSlider) {
                modelCoverSlider.addEventListener('input', () => {
                    syncModelCoverControl();
                    renderModelComparison();
                });
            }
            const illustrationSlider = document.getElementById('habitat-illustration-slider');
            if (illustrationSlider) {
                illustrationSlider.addEventListener('input', onIllustrationHabitatChange);
            }
            syncIllustrationHabitatControl();
        }

        function onIllustrationHabitatChange() {
            syncIllustrationHabitatControl();
            renderHabitatAlphaIllustration();
        }
        
        // ============================================================
        // RENDERING FUNCTIONS
        // ============================================================
        
        function renderSummaryStats() {
            if (!summaryData) return;
            const snapshotRows = summaryData.snapshot_results || [];
            const cumulativeRows = summaryData.cumulative_results || [];
            if (!snapshotRows.length || !cumulativeRows.length) return;

            const getBaseline = (valueType) => {
                const rows = snapshotRows.filter((r) => r.value_type === valueType);
                if (!rows.length) return 0;
                return Math.max(...rows.map((r) => Number(r.original_value_billions || 0)));
            };

            const baselineTourism = getBaseline('tourism');
            const baselineFisheries = getBaseline('fisheries');
            const baselineCoastal = getBaseline('coastal_protection');

            const annualAggregate = new Map();
            snapshotRows.forEach((r) => {
                const key = `${r.scenario}||${r.model}`;
                annualAggregate.set(
                    key,
                    (annualAggregate.get(key) || 0) + Number(r.total_loss_billions || 0)
                );
            });
            let worstAnnualLoss = 0;
            annualAggregate.forEach((value) => {
                if (value > worstAnnualLoss) worstAnnualLoss = value;
            });

            const cumulativeAggregate = new Map();
            cumulativeRows.forEach((r) => {
                const key = `${r.scenario}||${r.interpolation}||${r.model}`;
                cumulativeAggregate.set(
                    key,
                    (cumulativeAggregate.get(key) || 0) + Number(r.total_cumulative_loss_trillions || 0)
                );
            });
            let worstCumulativeLoss = 0;
            cumulativeAggregate.forEach((value) => {
                if (value > worstCumulativeLoss) worstCumulativeLoss = value;
            });

            const coralRow =
                cumulativeRows.find((r) => {
                    const scenario = (r.scenario || '').toLowerCase();
                    return scenario.includes('rcp85') && (r.period || '').includes('2100') && r.interpolation === 'linear';
                }) ||
                cumulativeRows.find((r) => (r.scenario || '').toLowerCase().includes('rcp85')) ||
                cumulativeRows[0];
            if (!coralRow) return;

            const html = `
                <div class="summary-row summary-row-top" style="text-align: center;">
                    <div class="stat-card">
                        <div class="stat-label">Coral Cover Change</div>
                        <div class="stat-value neutral">${Number(coralRow.cover_change_pp || 0).toFixed(1)}pp</div>
                        <div class="stat-detail">${Number(coralRow.baseline_cover_pct || 0).toFixed(1)}% → ${Number(coralRow.final_cover_pct || 0).toFixed(1)}%</div>
                    </div>
                </div>
                <div class="summary-row summary-row-middle">
                    <div class="stat-card">
                        <div class="stat-label">Baseline Tourism Value</div>
                        <div class="stat-value value">${formatBillions(baselineTourism)}</div>
                        <div class="stat-detail">Annual reef-associated tourism</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Baseline Coastal Protection Value</div>
                        <div class="stat-value value">${formatBillions(baselineCoastal)}</div>
                        <div class="stat-detail">Annual reef-derived flood protection</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Baseline Fisheries Value</div>
                        <div class="stat-value value">${formatBillions(baselineFisheries)}</div>
                        <div class="stat-detail">Annual reef fisheries value</div>
                    </div>
                </div>
                <div class="summary-row summary-row-bottom">
                    <div class="stat-card">
                        <div class="stat-label">Worst-Case End-point Loss</div>
                        <div class="stat-value loss">-${formatBillions(worstAnnualLoss)}</div>
                        <div class="stat-detail">Value lost by 2100 under worst-case emissions and depreciation scenario</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Worst-Case Aggregate Cumulative Loss</div>
                        <div class="stat-value loss">-$${Number(worstCumulativeLoss || 0).toFixed(2)}T</div>
                        <div class="stat-detail">2013 onward, summed across datasets</div>
                    </div>
                </div>
            `;
            
            document.getElementById('summary-stats').innerHTML = html;
        }
        
        function scenarioComparisonModelTag(modelKey) {
            if (modelKey === 'Tipping Point') return 'Tipping';
            return modelKey;
        }

        function renderScenarioComparison() {
            if (!summaryData) return;
            const selectedValueTypes = getScenarioComparisonValueTypes();
            const orderedValueTypes = VALUE_TYPES.filter((vt) => selectedValueTypes.includes(vt));

            if (orderedValueTypes.length === 0) {
                Plotly.newPlot(
                    'scenario-comparison-chart',
                    [],
                    {
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        font: { color: '#94a3b8', family: 'Instrument Sans' },
                        xaxis: { gridcolor: '#334155', title: 'Scenario' },
                        yaxis: { gridcolor: '#334155', title: 'Annual Loss ($ Billion)' },
                        annotations: [
                            {
                                x: 0.5,
                                y: 0.5,
                                xref: 'paper',
                                yref: 'paper',
                                text: 'Select at least one dataset',
                                showarrow: false,
                                font: { size: 16, color: '#94a3b8' },
                            },
                        ],
                        margin: { t: 20, r: 20, b: 100, l: 60 },
                    },
                    { responsive: true }
                );
                return;
            }

            const results = (summaryData.snapshot_results || [])
                .filter((r) => orderedValueTypes.includes(r.value_type));
            if (!results.length) return;

            const formatScenario = (s) => {
                const match = s.match(/rcp(\d+)_(\d+)/i);
                if (match) {
                    return `RCP ${match[1].charAt(0)}.${match[1].charAt(1)} — ${match[2]}`;
                }
                return s.replace('y_future_', '').replace(/_/g, ' ').toUpperCase();
            };

            const scenarios = sortScenarios([...new Set(results.map((r) => r.scenario))]);
            const models = sortModels([...new Set(results.map((r) => r.model))]);
            const scenarioLabels = scenarios.map((scenario) => formatScenario(scenario));

            const getLoss = (scenario, model, valueType) => {
                const row = results.find(
                    (r) =>
                        r.scenario === scenario &&
                        r.model === model &&
                        r.value_type === valueType
                );
                return row ? Number(row.total_loss_billions || 0) : 0;
            };

            const barWidth = 0.25;
            const intraGap = 0.04;
            const clusterGap = 0.55;
            const clusterInnerWidth = models.length * barWidth + (models.length - 1) * intraGap;
            const clusterStride = clusterInnerWidth + clusterGap;
            const barX = (index) => {
                const clusterIndex = Math.floor(index / models.length);
                const positionInCluster = index % models.length;
                const clusterStart = clusterIndex * clusterStride;
                return clusterStart + positionInCluster * (barWidth + intraGap);
            };

            const barSlots = scenarios.flatMap((scenario) =>
                models.map((model) => ({
                    scenario,
                    model,
                    modelKey: shortModelLabel(model),
                }))
            );
            const xValues = barSlots.map((_, index) => barX(index));
            const clusterCenters = scenarios.map(
                (_, scenarioIndex) => scenarioIndex * clusterStride + clusterInnerWidth / 2
            );
            const xMax = (scenarios.length - 1) * clusterStride + clusterInnerWidth - intraGap;

            const traces = orderedValueTypes.map((valueType) => ({
                x: xValues,
                y: barSlots.map((slot) => getLoss(slot.scenario, slot.model, valueType)),
                width: barWidth,
                customdata: barSlots.map((slot) => ({
                    scenario: formatScenario(slot.scenario),
                    model: slot.modelKey,
                    dataset: formatValueType(valueType),
                })),
                name: formatValueType(valueType),
                type: 'bar',
                hovertemplate:
                    '%{customdata.scenario}<br>' +
                    'Model: %{customdata.model}<br>' +
                    'Dataset: %{customdata.dataset}<br>' +
                    'Annual loss: $%{y:.1f}B<extra></extra>',
                marker: {
                    color: valueTypeColor(valueType),
                    line: { width: 0 },
                },
            }));

            const modelLabelAnnotations = barSlots.map((slot, index) => ({
                x: xValues[index],
                y: -0.05,
                xref: 'x',
                yref: 'paper',
                text: scenarioComparisonModelTag(slot.modelKey),
                showarrow: false,
                yanchor: 'top',
                font: { size: 10, color: '#94a3b8' },
            }));

            const scenarioLabelAnnotations = scenarios.map((scenario, scenarioIndex) => ({
                x: clusterCenters[scenarioIndex],
                y: -0.13,
                xref: 'x',
                yref: 'paper',
                text: scenarioLabels[scenarioIndex],
                showarrow: false,
                yanchor: 'top',
                font: { size: 12, color: '#cbd5e1' },
            }));

            const layout = {
                barmode: 'stack',
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: {
                    gridcolor: '#334155',
                    title: {
                        text: 'Climate scenario',
                        standoff: 42,
                    },
                    showticklabels: false,
                    range: [-barWidth * 0.5, xMax + barWidth * 0.5],
                },
                yaxis: {
                    gridcolor: '#334155',
                    title: 'Annual Loss ($ Billion)',
                },
                legend: {
                    orientation: 'h',
                    x: 0.5,
                    xanchor: 'center',
                    y: -0.34,
                    font: { size: 12 },
                },
                annotations: [...modelLabelAnnotations, ...scenarioLabelAnnotations],
                margin: { t: 20, r: 20, b: 132, l: 60 },
            };

            Plotly.newPlot('scenario-comparison-chart', traces, layout, { responsive: true });
        }
        
        function renderOverviewTrajectory(metric) {
            if (!trajectoryData) return;
            
            const modelFilter = document.getElementById('overview-model').value;
            const selectedValueTypes = getOverviewSelectedValueTypes();

            const rcpColors = {
                rcp45: CHART_COLORS.scenario?.rcp45 || '#3498db',
                rcp85: CHART_COLORS.scenario?.rcp85 || '#e74c3c',
            };
            // Linestyle = dataset type
            const datasetLineStyles = {
                tourism: 'solid',
                fisheries: 'dash',
                coastal_protection: 'dot',
            };

            // Map trace types for Plotly
            const metricMap = {
                'cumulative_loss': { title: 'Cumulative Loss ($ Trillion)' },
                'annual_loss': { title: 'Annual Loss ($ Billion/year)' }
            };
            const config = metricMap[metric];

            if (selectedValueTypes.length === 0) {
                Plotly.newPlot('trajectory-chart', [], {
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#94a3b8', family: 'Instrument Sans' },
                    xaxis: { gridcolor: '#334155', title: 'Year' },
                    yaxis: { gridcolor: '#334155', title: config.title },
                    annotations: [{
                        x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
                        text: 'Select at least one dataset',
                        showarrow: false,
                        font: { size: 16, color: '#94a3b8' },
                    }],
                    margin: { t: 20, r: 20, b: 80, l: 60 },
                }, { responsive: true });
                return;
            }

            // Filter data: use linear interpolation only, apply model filter
            let filtered = trajectoryData.filter(t => t.interpolation === 'linear');
            filtered = filtered.filter(t => selectedValueTypes.includes(t.value_type));
            filtered = [...filtered].sort(
                (a, b) => VALUE_TYPES.indexOf(a.value_type) - VALUE_TYPES.indexOf(b.value_type)
            );
            if (modelFilter !== 'all') {
                filtered = filtered.filter(t => {
                    const modelName = t.model.toLowerCase();
                    if (modelFilter === 'Linear') {
                        return modelName.includes('linear') && !modelName.includes('compound') && !modelName.includes('tipping');
                    } else if (modelFilter === 'Compound') {
                        return modelName.includes('compound') && !modelName.includes('tipping');
                    } else if (modelFilter === 'Tipping') {
                        return modelName.includes('tipping');
                    }
                    return t.model.includes(modelFilter);
                });
            }

            // Create traces
            const traces = filtered.map(t => {
                const scenario = t.scenario.toLowerCase();
                const color = rcpColors[scenario] || '#94a3b8';
                const linestyle = datasetLineStyles[t.value_type] || 'solid';
                
                let yData;
                if (metric === 'cumulative_loss') {
                    yData = t.cumulative_loss || [];
                } else if (metric === 'annual_loss') {
                    yData = t.annual_value_lost || t.annual_loss || [];
                }
                
                return {
                    x: t.years,
                    y: yData,
                    name: `${t.scenario.toUpperCase()} - ${formatValueType(t.value_type)}`,
                    showlegend: false,
                    mode: 'lines',
                    hovertemplate:
                        `${t.scenario.toUpperCase()} | ${formatValueType(t.value_type)}<br>` +
                        'Year: %{x}<br>' +
                        `${metric === 'cumulative_loss' ? 'Cumulative Loss: $%{y:.3f}T' : 'Annual Loss: $%{y:.2f}B/yr'}<extra></extra>`,
                    line: {
                        color: color,
                        dash: linestyle,
                        width: 2.5
                    }
                };
            });

            // Legend guide: colors indicate scenario, dash indicates dataset
            const legendGuideTraces = [
                {
                    x: [null], y: [null], mode: 'lines', name: 'RCP 4.5',
                    line: { color: rcpColors.rcp45, dash: 'solid', width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'RCP 8.5',
                    line: { color: rcpColors.rcp85, dash: 'solid', width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Tourism',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.tourism, width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Coastal protection',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.coastal_protection, width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Fisheries',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.fisheries, width: 3 },
                    hoverinfo: 'skip'
                },
            ];
            
            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { 
                    gridcolor: '#334155',
                    title: 'Year'
                },
                yaxis: { 
                    gridcolor: '#334155',
                    title: config.title,
                    // Adjust scale based on metric
                    type: metric === 'cumulative_loss' ? 'linear' : 'linear'
                },
                legend: { 
                    orientation: 'h', 
                    y: -0.25, // Move legend further down below plot, increasing space from x axis label
                    font: { size: 11 },
                },
                margin: { t: 20, r: 20, b: 80, l: 60 }
            };
            
            Plotly.newPlot('trajectory-chart', [...traces, ...legendGuideTraces], layout, {responsive: true});
        }
        
        function renderTrajectoryPage() {
            if (!trajectoryData) return;
            
            const interpolation = document.getElementById('traj-interpolation').value;
            const modelFilter = document.getElementById('traj-model').value;
            const selectedValueTypes = getTrajectorySelectedValueTypes();
            
            // Color = RCP scenario (blue for RCP45, red for RCP85) - matching scenario comparison chart
            const rcpColors = {
                'rcp45': CHART_COLORS.scenario?.rcp45 || '#3498db',
                'rcp85': CHART_COLORS.scenario?.rcp85 || '#e74c3c',
            };
            // Linestyle = dataset
            const datasetLineStyles = {
                tourism: 'solid',
                fisheries: 'dash',
                coastal_protection: 'dot',
            };
            
            // For coral cover, we only need unique scenario+interpolation combos (not per model)
            // since coral cover is the same for all economic models
            // Use a Set to get unique scenario+interpolation combos
            const seenCoralKeys = new Set();
            const coralFiltered = trajectoryData.filter(t => {
                const interpMatch = t.interpolation === interpolation;
                if (!interpMatch) return false;
                
                const key = `${t.scenario}_${t.interpolation}`;
                if (seenCoralKeys.has(key)) return false;
                seenCoralKeys.add(key);
                return true;
            });
            
            const coralTraces = coralFiltered.map(t => ({
                x: t.years,
                y: t.coral_cover,
                name: `${t.scenario.toUpperCase()}`,
                mode: 'lines',
                hovertemplate:
                    `${t.scenario.toUpperCase()}<br>` +
                    'Year: %{x}<br>' +
                    'Coral Cover: %{y:.1f}%<extra></extra>',
                line: { 
                    color: rcpColors[t.scenario.toLowerCase()] || '#94a3b8', 
                    width: 3 
                }
            }));
            
            Plotly.newPlot('coral-cover-chart', coralTraces, {
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { gridcolor: '#334155', title: 'Year' },
                yaxis: { gridcolor: '#334155', title: 'Coral Cover (%)' },
                legend: { orientation: 'h', y: -0.22, font: { size: 10 } },
                margin: { t: 20, r: 20, b: 90, l: 60 }
            }, {responsive: true});
            
            // For economic charts, filter by both interpolation and model
            let economicFiltered = trajectoryData.filter(t => t.interpolation === interpolation);
            if (modelFilter !== 'all') {
                economicFiltered = economicFiltered.filter(t => t.model.includes(modelFilter));
            }
            economicFiltered = economicFiltered.filter(t => selectedValueTypes.includes(t.value_type));
            economicFiltered = [...economicFiltered].sort(
                (a, b) => VALUE_TYPES.indexOf(a.value_type) - VALUE_TYPES.indexOf(b.value_type)
            );

            if (selectedValueTypes.length === 0) {
                const emptyLayout = (title) => ({
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#94a3b8', family: 'Instrument Sans' },
                    xaxis: { gridcolor: '#334155', title: 'Year' },
                    yaxis: { gridcolor: '#334155', title },
                    annotations: [
                        {
                            x: 0.5,
                            y: 0.5,
                            xref: 'paper',
                            yref: 'paper',
                            text: 'Select at least one dataset',
                            showarrow: false,
                            font: { size: 15, color: '#94a3b8' },
                        },
                    ],
                    margin: { t: 20, r: 20, b: 60, l: 60 },
                });
                Plotly.newPlot('annual-value-chart', [], emptyLayout('Annual Value ($ Billion)'), {responsive: true});
                Plotly.newPlot('cumulative-chart', [], emptyLayout('Cumulative Loss ($ Trillion)'), {responsive: true});
                return;
            }
            
            // Annual value chart - distinguish by model and scenario
            // Color = RCP scenario, Linestyle = Model
            const valueTraces = economicFiltered.map(t => {
                const scenario = t.scenario.toLowerCase();
                const color = rcpColors[scenario] || '#94a3b8';
                const dash = datasetLineStyles[t.value_type] || 'solid';
                
                return {
                    x: t.years,
                    y: t.annual_value,
                    name: `${t.scenario.toUpperCase()} - ${formatValueType(t.value_type)}`,
                    showlegend: false,
                    mode: 'lines',
                    hovertemplate:
                        `${t.scenario.toUpperCase()} | ${formatValueType(t.value_type)}<br>` +
                        'Year: %{x}<br>' +
                        'Annual Value: $%{y:.2f}B<extra></extra>',
                    line: { 
                        color: color,
                        dash: dash,
                        width: 2.5 
                    }
                };
            });

            const trajectoryLegendGuideTraces = [
                {
                    x: [null], y: [null], mode: 'lines', name: 'RCP 4.5',
                    line: { color: rcpColors.rcp45, dash: 'solid', width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'RCP 8.5',
                    line: { color: rcpColors.rcp85, dash: 'solid', width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Tourism',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.tourism, width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Coastal protection',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.coastal_protection, width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Fisheries',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.fisheries, width: 3 },
                    hoverinfo: 'skip'
                },
            ];
            
            Plotly.newPlot('annual-value-chart', [...valueTraces, ...trajectoryLegendGuideTraces], {
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { gridcolor: '#334155', title: 'Year' },
                yaxis: { gridcolor: '#334155', title: 'Annual Value ($ Billion)' },
                legend: { orientation: 'h', y: -0.28, font: { size: 10 } },
                margin: { t: 20, r: 20, b: 100, l: 60 }
            }, {responsive: true            });
            
            // Cumulative loss chart - cumulative sum of opportunity cost
            // Color = RCP scenario, Linestyle = Model
            const cumulativeTraces = economicFiltered.map(t => {
                const oppCost = t.annual_opportunity_cost || [];
                let cumulative = 0;
                const cumulativeOppCost = oppCost.length > 0
                    ? oppCost.map((val) => {
                        cumulative += Number(val || 0);
                        return cumulative / 1e3;
                    })
                    : (t.cumulative_loss || []);
                
                const scenario = t.scenario.toLowerCase();
                const color = rcpColors[scenario] || '#94a3b8';
                const dash = datasetLineStyles[t.value_type] || 'solid';
                
                return {
                    x: t.years,
                    y: cumulativeOppCost,
                    name: `${t.scenario.toUpperCase()} - ${formatValueType(t.value_type)}`,
                    showlegend: false,
                    mode: 'lines',
                    hovertemplate:
                        `${t.scenario.toUpperCase()} | ${formatValueType(t.value_type)}<br>` +
                        'Year: %{x}<br>' +
                        'Cumulative Loss: $%{y:.3f}T<extra></extra>',
                    line: { 
                        color: color,
                        dash: dash,
                        width: 2.5 
                    }
                };
            });
            
            Plotly.newPlot('cumulative-chart', [...cumulativeTraces, ...trajectoryLegendGuideTraces], {
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { gridcolor: '#334155', title: 'Year' },
                yaxis: { gridcolor: '#334155', title: 'Cumulative Loss ($ Trillion)' },
                legend: { orientation: 'h', y: -0.28, font: { size: 10 } },
                margin: { t: 20, r: 20, b: 100, l: 60 }
            }, {responsive: true});
        }
        
        function renderCountryChart() {
            const scenario = document.getElementById('country-scenario').value;
            const isCumulative = scenario.startsWith('cumulative_');
            const dataSource = isCumulative ? cumulativeCountryData : countryData;
            const valueType = getSelectedValueType('country-value-type', 'all');
            
            if (!dataSource) {
                console.warn('Country data not loaded. isCumulative:', isCumulative, 'dataSource:', !!dataSource);
                return;
            }
            
            const model = mapModelToCountryModel(document.getElementById('country-model').value);
            const limitValue = document.getElementById('country-limit').value;
            const metric = document.getElementById('country-metric').value;
            const colorMode = document.getElementById('country-color-mode').value;
            
            console.log('Rendering country chart:', { scenario, model, isCumulative, dataSourceLength: dataSource.length });
            
            // Match scenario exactly
            let filtered = dataSource.filter(c => 
                c.scenario === scenario && c.model === model
            );
            if (valueType !== 'all') {
                filtered = filtered.filter(c => c.value_type === valueType);
            } else {
                filtered = aggregateRowsByCountry(
                    filtered.map(row => ({ ...row, scenario, model })),
                    isCumulative,
                    true
                );
            }
            
            console.log('Filtered countries:', filtered.length, 'for scenario:', scenario, 'model:', model);
            
            // Sort by selected metric
            if (metric === 'value_loss') {
                const lossKey = isCumulative ? 'cumulative_loss' : 'value_loss';
                filtered = filtered.sort((a, b) => (b[lossKey] || 0) - (a[lossKey] || 0));
            } else if (metric === 'reef_loss_pct') {
                const fractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
                filtered = filtered.sort((a, b) => (b[fractionKey] || 0) - (a[fractionKey] || 0));
            } else if (metric === 'gdp_pct' && !isCumulative && window.gdpImpactLookup) {
                // Sort by GDP % if available
                filtered = filtered.sort((a, b) => {
                    const valA = valueType === 'all'
                        ? (a.loss_as_gdp_pct || 0)
                        : ((window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${a.country}`] || {}).loss_as_gdp_pct || 0);
                    const valB = valueType === 'all'
                        ? (b.loss_as_gdp_pct || 0)
                        : ((window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${b.country}`] || {}).loss_as_gdp_pct || 0);
                    return valB - valA;
                });
            } else {
                // Fallback: sort by reef loss fraction
                const fractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
                filtered = filtered.sort((a, b) => (b[fractionKey] || 0) - (a[fractionKey] || 0));
            }
            
            // Apply limit
            if (limitValue !== 'all') {
                filtered = filtered.slice(0, parseInt(limitValue));
            }
            
            // Dynamic height based on number of countries
            const chartHeight = Math.max(400, filtered.length * 25);
            document.getElementById('country-chart').style.height = chartHeight + 'px';
            
            // Configure based on metric (x-axis)
            let xData, xTitle, textFn;
            if (metric === 'value_loss') {
                const lossKey = isCumulative ? 'cumulative_loss' : 'value_loss';
                xData = filtered.map(c => (c[lossKey] || 0) / 1e6);
                xTitle = isCumulative ? 'Cumulative Loss ($ Million)' : 'Value Loss ($ Million)';
                textFn = c => `$${((c[lossKey] || 0) / 1e6).toFixed(1)}M`;
            } else if (metric === 'reef_loss_pct') {
                const fractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
                xData = filtered.map(c => (c[fractionKey] || 0) * 100);
                xTitle = isCumulative ? 'Cumulative Loss Fraction (%)' : 'Loss Fraction (%)';
                textFn = c => `${((c[fractionKey] || 0) * 100).toFixed(1)}%`;
            } else if (metric === 'gdp_pct') {
                // For GDP %, need to look up from gdpImpactData
                if (!isCumulative && window.gdpImpactLookup) {
                    xData = filtered.map(c => {
                        if (valueType === 'all') return c.loss_as_gdp_pct || 0;
                        const rec = window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${c.country}`];
                        return rec ? rec.loss_as_gdp_pct : 0;
                    });
                    xTitle = 'Loss as % of National GDP';
                    textFn = c => {
                        if (valueType === 'all') return `${(c.loss_as_gdp_pct || 0).toFixed(2)}%`;
                        const rec = window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${c.country}`];
                        return rec ? `${rec.loss_as_gdp_pct.toFixed(2)}%` : '0%';
                    };
                } else {
                    // Fallback for cumulative or if GDP data not available
                    const fractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
                    xData = filtered.map(c => (c[fractionKey] || 0) * 100);
                    xTitle = isCumulative ? 'Cumulative Loss Fraction (%)' : 'Loss Fraction (%)';
                    textFn = c => `${((c[fractionKey] || 0) * 100).toFixed(1)}%`;
                }
            }

            // Configure colouring (loss % of reef tourism, % of national GDP, or absolute value loss)
            let colorValues;
            let colorbarTitle;
            let colorbarFormat;

            const reefFractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
            const lossKey = isCumulative ? 'cumulative_loss' : 'value_loss';

            if (colorMode === 'gdp_pct' && !isCumulative && window.gdpImpactLookup) {
                // Use loss_as_gdp_pct from gdpImpactData where available (annual scenarios only)
                colorValues = filtered.map(c => {
                    if (valueType === 'all') return c.loss_as_gdp_pct || 0;
                    const rec = window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${c.country}`];
                    return rec ? rec.loss_as_gdp_pct : 0;
                });
                colorbarTitle = '% GDP';
                colorbarFormat = '.1f';
            } else if (colorMode === 'value_loss') {
                // Color by absolute value loss
                colorValues = filtered.map(c => (c[lossKey] || 0) / 1e6); // Convert to millions for color scale
                colorbarTitle = isCumulative ? 'Cumulative Loss ($M)' : 'Value Loss ($M)';
                colorbarFormat = ',.0f';
            } else {
                // Default: colour by reef tourism loss fraction
                colorValues = filtered.map(c => c[reefFractionKey] || 0);
                colorbarTitle = isCumulative ? 'Cumulative Loss %' : 'Loss %';
                colorbarFormat = '.0%';
            }
            
            const trace = {
                y: filtered.map(c => c.country),
                x: xData,
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: colorValues,
                    colorscale: lossColorscale([0.25, 0.5]),
                    colorbar: {
                        title: colorbarTitle,
                        tickformat: colorbarFormat
                    }
                },
                text: filtered.map(textFn),
                textposition: 'outside',
                hovertemplate:
                    'Country: %{y}<br>' +
                    `${xTitle}: %{x:.2f}<extra></extra>`
            };
            
            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { 
                    gridcolor: '#334155',
                    title: xTitle
                },
                yaxis: { 
                    autorange: 'reversed'
                },
                margin: { t: 20, r: 100, b: 60, l: 150 }
            };
            
            Plotly.newPlot('country-chart', [trace], layout, {responsive: true});
        }
        
        function renderGdpChart() {
            if (!gdpImpactData) return;
            
            const scenario = document.getElementById('gdp-scenario').value;
            const model = document.getElementById('gdp-model').value;
            const limitValue = document.getElementById('gdp-limit').value;
            const metric = document.getElementById('gdp-metric').value;
            
            // Match scenario exactly
            let filtered = gdpImpactData.filter(c => 
                c.scenario === scenario && c.model === model
            );
            
            // Filter and sort by selected metric
            if (metric === 'loss_as_gdp_pct') {
                filtered = filtered.filter(c => c.loss_as_gdp_pct > 0)
                    .sort((a, b) => b.loss_as_gdp_pct - a.loss_as_gdp_pct);
            } else {
                filtered = filtered.filter(c => c.value_loss > 0)
                    .sort((a, b) => b.value_loss - a.value_loss);
            }
            
            // Apply limit
            if (limitValue !== 'all') {
                filtered = filtered.slice(0, parseInt(limitValue));
            }
            
            // Dynamic height based on number of countries
            const chartHeight = Math.max(500, filtered.length * 30);
            document.getElementById('gdp-chart').style.height = chartHeight + 'px';
            
            // Configure based on metric
            let xData, xTitle, textFn, colorbarTitle, colorbarFormat;
            if (metric === 'loss_as_gdp_pct') {
                xData = filtered.map(c => c.loss_as_gdp_pct);
                xTitle = 'Projected Loss as % of National GDP';
                textFn = c => `${c.loss_as_gdp_pct.toFixed(2)}%`;
                colorbarTitle = '% GDP';
                colorbarFormat = '.1f';
            } else {
                xData = filtered.map(c => c.value_loss / 1e6);
                xTitle = 'Value Loss ($ Million)';
                textFn = c => `$${(c.value_loss / 1e6).toFixed(1)}M`;
                colorbarTitle = 'Loss $M';
                colorbarFormat = '.0f';
            }
            
            const trace = {
                y: filtered.map(c => c.country),
                x: xData,
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: filtered.map(c => c.loss_as_gdp_pct),
                    colorscale: lossColorscale([0.1, 0.3]),
                    colorbar: {
                        title: colorbarTitle,
                        tickformat: colorbarFormat
                    }
                },
                text: filtered.map(textFn),
                textposition: 'outside',
                hovertemplate:
                    'Country: %{y}<br>' +
                    `${xTitle}: %{x:.2f}<extra></extra>`
            };
            
            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { 
                    gridcolor: '#334155',
                    title: xTitle
                },
                yaxis: { 
                    autorange: 'reversed'
                },
                margin: { t: 20, r: 100, b: 0, l: 150 }
            };
            
            Plotly.newPlot('gdp-chart', [trace], layout, {responsive: true});
        }
        
        function renderGdpComparison() {
            if (!gdpImpactData) return;
            
            const model = mapModelToCountryModel(document.getElementById('gdp-comparison-model').value);
            const metric = document.getElementById('gdp-comparison-metric').value;
            const limitValue = document.getElementById('gdp-comparison-limit').value;
            const valueType = getSelectedValueType('gdp-value-type', 'all');
            const limit = limitValue === 'all' ? 9999 : parseInt(limitValue);
            let gdpSource = gdpImpactData.filter(c => c.model === model);
            if (valueType !== 'all') {
                gdpSource = gdpSource.filter(c => c.value_type === valueType);
            } else {
                const grouped = new Map();
                gdpSource.forEach((row) => {
                    const key = `${row.scenario}||${row.model}||${row.country}`;
                    if (!grouped.has(key)) {
                        grouped.set(key, {
                            ...row,
                            value_type: 'all',
                            value_loss: 0,
                            loss_as_gdp_pct: 0,
                        });
                    }
                    const g = grouped.get(key);
                    g.value_loss += Number(row.value_loss || 0);
                    const nationalGdp = Number(row.national_gdp || 0);
                    if (nationalGdp > 0) {
                        g.loss_as_gdp_pct = (g.value_loss / nationalGdp) * 100;
                    }
                });
                gdpSource = Array.from(grouped.values());
            }
            
            // Get top countries by worst-case impact for selected model and metric
            const worstCase = gdpSource.filter(c => 
                c.scenario.includes('rcp85') && c.scenario.includes('2100') &&
                c.model === model
            );
            
            // Sort by selected metric
            const metricKey = metric === 'loss_as_gdp_pct' ? 'loss_as_gdp_pct' : 'value_loss';
            worstCase.sort((a, b) => (b[metricKey] || 0) - (a[metricKey] || 0));
            const topCountries = worstCase.slice(0, limit).map(c => c.country);
            
            // Dynamic height based on number of countries - increased spacing
            const chartHeight = Math.max(1000, topCountries.length * 100);
            document.getElementById('gdp-comparison-chart').style.height = chartHeight + 'px';
            
            // Create traces: 2050 overlaid on 2100 for each RCP (2 bars per country)
            // 2100 is the full bar (lighter), 2050 is overlaid on top (darker)
            // Use custom y positions to group RCP 4.5 and RCP 8.5 side by side
            const rcpScenarios = [
                { rcp: 'rcp45', color: scenarioColor('rcp45'), name: 'RCP 4.5', yOffset: -0.2 },
                { rcp: 'rcp85', color: scenarioColor('rcp85'), name: 'RCP 8.5', yOffset: 0.2 },
            ];
            
            const traces = [];
            
            // Create custom y positions for each country to offset RCP bars
            // Use larger spacing (multiply by 1.5) to spread countries out more
            const createYPositions = (offset) => {
                return topCountries.map((country, idx) => {
                    // Use numeric index with larger spacing and offset to create side-by-side bars
                    return idx * 1.5 + offset;
                });
            };
            
            // For each RCP, create two traces: 2100 (base, lighter) and 2050 (overlay, darker)
            rcpScenarios.forEach(rcp => {
                // Get data for both years
                const data2100 = gdpSource.filter(c => 
                    c.scenario.toLowerCase().includes(rcp.rcp) && 
                    c.scenario.includes('2100') &&
                    c.model === model &&
                    topCountries.includes(c.country)
                );
                
                const data2050 = gdpSource.filter(c => 
                    c.scenario.toLowerCase().includes(rcp.rcp) && 
                    c.scenario.includes('2050') &&
                    c.model === model &&
                    topCountries.includes(c.country)
                );
                
                const countryMap2100 = {};
                data2100.forEach(d => { countryMap2100[d.country] = d[metricKey] || 0; });
                
                const countryMap2050 = {};
                data2050.forEach(d => { countryMap2050[d.country] = d[metricKey] || 0; });
                
                const yPositions = createYPositions(rcp.yOffset);
                
                // 2100 trace (base, lighter opacity) - full height
                traces.push({
                    y: yPositions,
                    x: topCountries.map(c => countryMap2100[c] || 0),
                    customdata: topCountries,
                    name: `${rcp.name} - 2100`,
                    type: 'bar',
                    orientation: 'h',
                    hovertemplate:
                        'Country: %{customdata}<br>' +
                        '%{fullData.name}<br>' +
                        `${metric === 'loss_as_gdp_pct' ? 'Projected Loss as GDP: %{x:.2f}%' : 'Projected Value Loss: $%{x:,.0f}'}` +
                        '<extra></extra>',
                    width: 0.4,  // Narrower bars
                    marker: {
                        color: rcp.color,
                        opacity: 0.5,  // Lighter for base
                        line: { width: 0 }
                    }
                });
                
                // 2050 trace (overlay, darker opacity) - overlaid on top of 2100
                // Since 2050 < 2100, this will show as a darker section on the lighter bar
                traces.push({
                    y: yPositions,
                    x: topCountries.map(c => countryMap2050[c] || 0),
                    customdata: topCountries,
                    name: `${rcp.name} - 2050`,
                    type: 'bar',
                    orientation: 'h',
                    hovertemplate:
                        'Country: %{customdata}<br>' +
                        '%{fullData.name}<br>' +
                        `${metric === 'loss_as_gdp_pct' ? 'Projected Loss as GDP: %{x:.2f}%' : 'Projected Value Loss: $%{x:,.0f}'}` +
                        '<extra></extra>',
                    width: 0.4,  // Narrower bars
                    marker: {
                        color: rcp.color,
                        opacity: 1.0,  // Darker for overlay
                        line: { width: 0 }
                    }
                });
            });
            
            // Determine axis title based on metric
            const xAxisTitle = metric === 'loss_as_gdp_pct' 
                ? 'Projected Loss as % of National GDP'
                : 'Projected Value Loss ($ Million)';
            
            // Format function for text labels
            const formatValue = (val) => {
                if (metric === 'loss_as_gdp_pct') {
                    return `${val.toFixed(2)}%`;
                } else {
                    if (val >= 1e6) return `$${(val / 1e6).toFixed(1)}M`;
                    if (val >= 1e3) return `$${(val / 1e3).toFixed(1)}k`;
                    return `$${val.toFixed(0)}`;
                }
            };
            
            // Create custom y-axis tick labels (country names at their base positions)
            // Match the spacing used in createYPositions
            const yTickPositions = topCountries.map((_, idx) => idx * 1.5);
            const yTickLabels = topCountries;
            
            const layout = {
                barmode: 'overlay',  // Overlay 2050 on 2100 within each RCP
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { 
                    gridcolor: '#334155',
                    title: xAxisTitle
                },
                yaxis: { 
                    tickmode: 'array',
                    tickvals: yTickPositions,
                    ticktext: yTickLabels,
                    autorange: 'reversed'
                },
                legend: { 
                    orientation: 'h', 
                    y: -0.15,
                    font: { size: 11 }
                },
                margin: { t: 20, r: 20, b: 80, l: 150 }
            };
            
            Plotly.newPlot('gdp-comparison-chart', traces, layout, {responsive: true});
        }
        
        function getModelCurvesPayload() {
            if (!modelCurves) return { curves: {}, metadata: null };
            if (modelCurves.curves) {
                return { curves: modelCurves.curves, metadata: modelCurves.metadata || null };
            }
            return { curves: modelCurves, metadata: null };
        }

        const MODEL_DELTA_AXIS_TITLE = 'Change in coral cover (ΔC<sub>pp</sub>)';
        const MODEL_HOVER_DELTA = 'ΔC<sub>pp</sub>: %{x:.1f}';
        const CHEN_PAPER_URL = 'https://doi.org/10.1016/j.gloenvcha.2014.10.011';
        const CHEN_TOURISM_ELASTICITY = 3.8069;
        const COMPOUND_RATE = 0.0381;
        const TIPPING_THRESHOLD = 0.10;
        const TIPPING_PRE_RATE = 0.0381;

        function getModelInitialCoverPct(metadata) {
            const slider = document.getElementById('model-initial-cover');
            if (slider && slider.value !== '') {
                return Number(slider.value);
            }
            return metadata?.reference_cover_pct ?? 35;
        }

        function syncModelCoverControl() {
            const output = document.getElementById('model-initial-cover-value');
            const slider = document.getElementById('model-initial-cover');
            const coastalC0 = document.getElementById('coastal-c0-text');
            const pct = getModelInitialCoverPct();
            if (output) {
                output.textContent = `${pct}%`;
            }
            if (coastalC0) {
                coastalC0.textContent = String(pct);
            }
            if (slider) {
                slider.setAttribute('aria-valuenow', String(pct));
            }
        }

        function getModelDeltaPpRange(curves) {
            const compound = curves?.compound;
            if (compound?.delta_cc?.length) {
                return compound.delta_cc;
            }
            const values = [];
            for (let x = -50; x <= 10; x += 0.5) {
                values.push(x);
            }
            return values;
        }

        function computeHabitatRemaining(deltaPp, initialCoverPct, habitatFloorA) {
            const c0 = initialCoverPct / 100;
            const delta = deltaPp / 100;
            if (c0 <= 0) {
                return 0;
            }
            const cFinal = Math.max(c0 + delta, 0);
            const multiplier = habitatFloorA + (1 - habitatFloorA) * (cFinal / c0);
            return Math.max(0, 100 * multiplier);
        }

        function computeChenRemaining(deltaPp, initialCoverPct, sector) {
            const c0 = initialCoverPct / 100;
            const delta = deltaPp / 100;
            if (c0 <= 0) {
                return 0;
            }
            if (sector === 'fisheries') {
                return computeHabitatRemaining(deltaPp, initialCoverPct, getHabitatExportA());
            }
            const relative = delta / c0;
            const fracChange = sector === 'tourism'
                ? relative * CHEN_TOURISM_ELASTICITY
                : relative;
            return Math.max(0, 100 * (1 + fracChange));
        }

        function computeCoastalRemaining(deltaPp, initialCoverPct) {
            const c0 = initialCoverPct / 100;
            const cFinal = Math.max(c0 + deltaPp / 100, 0);
            if (c0 <= 0) {
                return 0;
            }
            const ratio = Math.min(cFinal / c0, 1);
            return Math.max(0, 100 * ratio);
        }

        function computeCoastalCurve(deltaPpArray, initialCoverPct) {
            return deltaPpArray.map((deltaPp) =>
                computeCoastalRemaining(deltaPp, initialCoverPct)
            );
        }

        function computeChenCurve(deltaPpArray, initialCoverPct, sector) {
            return deltaPpArray.map((deltaPp) =>
                computeChenRemaining(deltaPp, initialCoverPct, sector)
            );
        }

        function computeCompoundRemaining(deltaPp) {
            if (deltaPp >= 0) {
                return 100;
            }
            return Math.max(0, 100 * Math.pow(1 - COMPOUND_RATE, Math.abs(deltaPp)));
        }

        function computeCompoundCurve(deltaPpArray) {
            return deltaPpArray.map(computeCompoundRemaining);
        }

        function computeTippingRemaining(deltaPp, initialCoverPct) {
            const c0 = initialCoverPct / 100;
            const delta = deltaPp / 100;
            let remaining = deltaPp < 0
                ? 100 * Math.pow(1 - TIPPING_PRE_RATE, Math.abs(deltaPp))
                : 100;
            const remainingCc = Math.max(c0 + delta, 0);
            if (remainingCc < TIPPING_THRESHOLD) {
                remaining *= 0;
            }
            return Math.max(0, remaining);
        }

        function computeTippingCurve(deltaPpArray, initialCoverPct) {
            return deltaPpArray.map((deltaPp) =>
                computeTippingRemaining(deltaPp, initialCoverPct)
            );
        }

        function findTourismCliffPp(initialCoverPct) {
            return -initialCoverPct / CHEN_TOURISM_ELASTICITY;
        }

        function findTippingCliffPp(initialCoverPct) {
            return -(initialCoverPct - TIPPING_THRESHOLD * 100);
        }

        function getChenChartXRange(initialCoverPct) {
            const tourismCliff = Math.abs(findTourismCliffPp(initialCoverPct));
            const xMin = -Math.min(
                55,
                Math.max(12, Math.ceil(tourismCliff) + 8, Math.min(initialCoverPct, 30))
            );
            return [xMin, 5];
        }

        function chenPaperLink(label = 'Chen et al. (2015)') {
            return `<a href="${CHEN_PAPER_URL}" target="_blank" rel="noopener noreferrer">${label}</a>`;
        }

        function linkChenCitations(text) {
            const raw = String(text);
            if (raw.includes('<a ') || raw.includes(CHEN_PAPER_URL)) {
                return raw;
            }
            return raw.replace(
                /Chen et al\.(?:\s*\(\d{4}(?:\/\d{4})?\))?/g,
                (match) => chenPaperLink(match)
            );
        }

        function plotModelChart(elementId, traces, layout) {
            const el = document.getElementById(elementId);
            if (!el) {
                return;
            }
            const plotFn = el.data ? Plotly.react : Plotly.newPlot;
            plotFn(elementId, traces, layout, { responsive: true });
        }

        function renderHabitatAlphaIllustration() {
            const chartEl = document.getElementById('habitat-alpha-chart');
            if (!chartEl || !summaryData) return;

            syncIllustrationHabitatControl();
            const selectedAlpha = getIllustrationHabitatA();
            const rows = (summaryData.snapshot_results || []).filter(
                (r) => r.value_type === 'fisheries' && isLinearModelName(r.model)
            );
            if (!rows.length) return;

            const formatScenario = (s) => {
                const match = s.match(/rcp(\d+)_(\d+)/i);
                if (match) {
                    return `RCP ${match[1].charAt(0)}.${match[1].charAt(1)} — ${match[2]}`;
                }
                return s;
            };

            const alphaValues = HABITAT_ILLUSTRATION_OPTIONS;
            const scenarios = sortScenarios([...new Set(rows.map((r) => r.scenario))]);
            const exportA = getHabitatExportA();

            const traces = scenarios.map((scenario) => ({
                x: alphaValues,
                y: alphaValues.map((alpha) => {
                    const row = rows.find((r) => r.scenario === scenario);
                    return row ? scaleFisheriesLinearLoss(row.total_loss_billions, alpha) : 0;
                }),
                name: formatScenario(scenario),
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: scenarioColor(scenario), width: 2.5 },
                marker: { size: 7 },
                hovertemplate:
                    `${formatScenario(scenario)}<br>` +
                    'α = %{x}<br>' +
                    'Fisheries annual loss: $%{y:.2f}B<extra></extra>',
            }));

            traces.push({
                x: [selectedAlpha, selectedAlpha],
                y: [0, Math.max(
                    ...alphaValues.map((alpha) =>
                        rows.reduce(
                            (sum, row) => sum + scaleFisheriesLinearLoss(row.total_loss_billions, alpha),
                            0
                        )
                    )
                ) * 1.05],
                name: `Selected α = ${selectedAlpha}`,
                type: 'scatter',
                mode: 'lines',
                line: { color: valueTypeColor('fisheries'), width: 2, dash: 'dash' },
                hoverinfo: 'skip',
                showlegend: true,
            });

            const totalAtSelected = rows.reduce(
                (sum, row) => sum + scaleFisheriesLinearLoss(row.total_loss_billions, selectedAlpha),
                0
            );

            Plotly.newPlot('habitat-alpha-chart', traces, {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: {
                    gridcolor: '#334155',
                    title: 'Habitat floor α',
                    tickmode: 'array',
                    tickvals: alphaValues,
                    ticktext: alphaValues.map(String),
                    range: [-0.05, 1.05],
                },
                yaxis: {
                    gridcolor: '#334155',
                    title: {
                        text: 'Fisheries annual loss<br>($ Billion, linear model)',
                        standoff: 18,
                    },
                    automargin: true,
                    range: [0, 4],
                    fixedrange: true,
                },
                legend: { orientation: 'h', y: -0.22, font: { size: 11 } },
                margin: { t: 20, r: 20, b: 90, l: 80 },
                annotations: [
                    {
                        x: selectedAlpha,
                        y: totalAtSelected,
                        text: `$${totalAtSelected.toFixed(2)}B total<br>at α = ${selectedAlpha}`,
                        showarrow: true,
                        arrowhead: 2,
                        ax: 40,
                        ay: -30,
                        font: { size: 11, color: valueTypeColor('fisheries') },
                    },
                    ...(exportA !== selectedAlpha ? [] : [{
                        x: exportA,
                        y: 1.1,
                        xref: 'x',
                        yref: 'paper',
                        text: 'Analysis default α = 0.4',
                        showarrow: false,
                        yanchor: 'top',
                        font: { size: 10, color: '#64748b' },
                    }]),
                ],
            }, { responsive: true });
        }

        function renderModelComparison() {
            const { curves, metadata } = getModelCurvesPayload();
            if (!curves || Object.keys(curves).length === 0) return;

            const refCover = getModelInitialCoverPct(metadata);
            syncModelCoverControl();

            const subtitleEl = document.getElementById('model-chart-subtitle');
            if (subtitleEl) {
                subtitleEl.innerHTML =
                    `${chenPaperLink()} tourism elasticity and compound sensitivity at C<sub>0</sub> = ${refCover}%.`;
            }

            const deltaPp = getModelDeltaPpRange(curves);
            const tourismRemaining = computeChenCurve(deltaPp, refCover, 'tourism');
            const compoundRemaining = computeCompoundCurve(deltaPp);

            const tourismTraces = [
                {
                    x: deltaPp,
                    y: tourismRemaining,
                    name: 'Linear — tourism (3.81%/pp)',
                    hovertemplate:
                        'Linear — tourism (3.81%/pp)<br>' +
                        `${MODEL_HOVER_DELTA}<br>` +
                        'Remaining value: %{y:.1f}%<extra></extra>',
                    mode: 'lines',
                    line: { color: valueTypeColor('tourism'), width: 3, dash: 'solid' },
                },
                {
                    x: deltaPp,
                    y: compoundRemaining,
                    name: 'Compound (3.81%/pp of previous value)',
                    hovertemplate:
                        'Compound (3.81%/pp of previous value)<br>' +
                        `${MODEL_HOVER_DELTA}<br>` +
                        'Remaining value: %{y:.1f}%<extra></extra>',
                    mode: 'lines',
                    line: { color: modelColor('Compound'), width: 3, dash: 'solid' },
                },
            ];

            const tourismZeroX = findTourismCliffPp(refCover);
            const coverZeroX = -refCover;
            const [chenXMin, chenXMax] = getChenChartXRange(refCover);

            plotModelChart('model-chart', tourismTraces, {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                autosize: true,
                title: {
                    text: 'Tourism — linear & compound',
                    font: { size: 14, color: '#e2e8f0' },
                    x: 0.5,
                },
                xaxis: {
                    gridcolor: '#334155',
                    title: MODEL_DELTA_AXIS_TITLE,
                    zeroline: true,
                    zerolinecolor: '#64748b',
                    range: [chenXMin, chenXMax],
                },
                yaxis: {
                    gridcolor: '#334155',
                    title: metadata?.y_axis || 'Remaining value (% of baseline)',
                    range: [0, 105],
                },
                legend: { orientation: 'h', y: -0.18 },
                margin: { t: 50, r: 30, b: 80, l: 60 },
                shapes: [
                    { type: 'line', x0: 0, x1: 0, y0: 0, y1: 105,
                      line: { color: '#64748b', width: 1, dash: 'dot' } },
                    { type: 'line', x0: coverZeroX, x1: coverZeroX, y0: 0, y1: 105,
                      line: { color: '#475569', width: 1, dash: 'longdash' } },
                    { type: 'line', x0: tourismZeroX, x1: tourismZeroX, y0: 0, y1: 105,
                      line: { color: valueTypeColor('tourism'), width: 1, dash: 'dash' } },
                ],
                annotations: [
                    { x: coverZeroX, y: 102, text: `C<sub>0</sub>→0`,
                      showarrow: false, font: { color: '#64748b', size: 9 },
                      xanchor: 'right', xshift: -4 },
                    { x: tourismZeroX, y: 55,
                      text: `Tourism value zero<br>(${tourismZeroX.toFixed(1)} pp)`,
                      showarrow: true, arrowhead: 2, arrowsize: 0.8,
                      arrowcolor: valueTypeColor('tourism'), ax: -35, ay: 0,
                      font: { color: valueTypeColor('tourism'), size: 9 } },
                ],
            });

            const TP_COLORS = {
                '0.15': '#ef4444',
                '0.25': '#f97316',
                '0.4': valueTypeColor('coastal_protection'),
                '0.6': '#38bdf8',
            };

            const tippingPointTraces = Object.entries(curves)
                .filter(([key]) => key.startsWith('tipping_point_'))
                .sort(([, a], [, b]) => (a.original_cc || 0) - (b.original_cc || 0))
                .map(([key, curve]) => {
                    const ogCc = curve.original_cc || 0.5;
                    const ogKey = String(ogCc);
                    const color = TP_COLORS[ogKey]
                        ?? `hsl(${Math.round(200 - ogCc * 180)}, 70%, 55%)`;
                    return {
                        x: curve.delta_cc,
                        y: curve.remaining_value,
                        name: curve.name,
                        hovertemplate:
                            `${curve.name}<br>` +
                            `${MODEL_HOVER_DELTA}<br>` +
                            'Remaining value: %{y:.1f}%<extra></extra>',
                        mode: 'lines',
                        opacity: 0.45,
                        line: { color, width: 2, dash: 'dot' },
                    };
                });

            const selectedTipping = computeTippingCurve(deltaPp, refCover);
            const selectedCliffX = findTippingCliffPp(refCover);
            tippingPointTraces.push({
                x: deltaPp,
                y: selectedTipping,
                name: `Selected C₀ (${refCover}%)`,
                hovertemplate:
                    `Selected C₀ (${refCover}%)<br>` +
                    `${MODEL_HOVER_DELTA}<br>` +
                    'Remaining value: %{y:.1f}%<extra></extra>',
                mode: 'lines',
                line: { color: '#f8fafc', width: 4 },
            });

            const tippingPointMarkers = [
                {
                    x: [selectedCliffX],
                    y: [50],
                    hovertemplate:
                        `Selected C₀ — threshold crossed<br>` +
                        `${MODEL_HOVER_DELTA}<extra></extra>`,
                    mode: 'markers',
                    marker: {
                        symbol: 'line-ns',
                        size: 16,
                        color: '#f8fafc',
                        line: { width: 2, color: '#f8fafc' },
                    },
                    showlegend: false,
                },
            ];

            const tippingPointCliffShapes = [
                { type: 'line', x0: 0, x1: 0, y0: 0, y1: 105,
                  line: { color: '#64748b', width: 1, dash: 'dot' } },
                { type: 'line', x0: selectedCliffX, x1: selectedCliffX, y0: 0, y1: 105,
                  line: { color: 'rgba(248,250,252,0.45)', width: 1.5, dash: 'dash' } },
            ];

            const tippingPointLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                autosize: true,
                title: {
                    text: 'Tipping-point sensitivity (all sectors)',
                    font: { size: 14, color: '#e2e8f0' },
                    x: 0.5,
                },
                xaxis: {
                    gridcolor: '#334155',
                    title: MODEL_DELTA_AXIS_TITLE,
                    zeroline: true,
                    zerolinecolor: '#64748b',
                    range: [-55, 5],
                },
                yaxis: {
                    gridcolor: '#334155',
                    title: metadata?.y_axis || 'Remaining value (% of baseline)',
                    range: [0, 105],
                },
                legend: { orientation: 'h', y: -0.18 },
                margin: { t: 50, r: 30, b: 80, l: 60 },
                shapes: tippingPointCliffShapes,
                annotations: [
                    { x: -27, y: 102, xanchor: 'center', showarrow: false,
                      text: 'System collapse threshold: 10% cover', font: { color: '#64748b', size: 9 } },
                ],
            };

            plotModelChart(
                'tipping-point-chart',
                [...tippingPointTraces, ...tippingPointMarkers],
                tippingPointLayout
            );

            renderCoastalDepreciationChart(refCover, metadata);
            renderHabitatAlphaIllustration();
        }

        function renderCoastalDepreciationChart(refCover, metadata) {
            const chartEl = document.getElementById('coastal-model-chart');
            if (!chartEl) return;

            const { curves } = getModelCurvesPayload();
            if (!curves || Object.keys(curves).length === 0) return;

            const deltaPp = getModelDeltaPpRange(curves);
            const coastalRemaining = computeCoastalCurve(deltaPp, refCover);
            const coverZeroX = -refCover;

            plotModelChart('coastal-model-chart', [{
                x: deltaPp,
                y: coastalRemaining,
                name: 'Cover-proportional — coastal protection',
                hovertemplate:
                    'Cover-proportional — coastal protection<br>' +
                    `${MODEL_HOVER_DELTA}<br>` +
                    'Remaining value: %{y:.1f}%<extra></extra>',
                mode: 'lines',
                line: { color: valueTypeColor('coastal_protection'), width: 3 },
            }], {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: {
                    gridcolor: '#334155',
                    title: MODEL_DELTA_AXIS_TITLE,
                    zeroline: true,
                    zerolinecolor: '#64748b',
                    range: [-55, 5],
                },
                yaxis: {
                    gridcolor: '#334155',
                    title: metadata?.y_axis || 'Remaining value (% of baseline)',
                    range: [0, 105],
                },
                legend: { orientation: 'h', y: -0.15 },
                margin: { t: 20, r: 20, b: 70, l: 60 },
                shapes: [
                    { type: 'line', x0: 0, x1: 0, y0: 0, y1: 105,
                      line: { color: '#64748b', width: 1, dash: 'dot' } },
                    { type: 'line', x0: coverZeroX, x1: coverZeroX, y0: 0, y1: 105,
                      line: { color: '#475569', width: 1, dash: 'longdash' } },
                ],
                annotations: [
                    { x: coverZeroX, y: 102, text: `C<sub>0</sub>→0`,
                      showarrow: false, font: { color: '#64748b', size: 9 },
                      xanchor: 'right', xshift: -4 },
                ],
            });
        }

        // ============================================================
        // METHODS — ENVIRONMENTAL COVARIATE VISUALIZATIONS
        // ============================================================

        const ENV_MAP_TARGETS = {
            historical: { selectId: 'env-metric-historical', chartId: 'env-map-historical' },
            recent: { selectId: 'env-metric-recent', chartId: 'env-map-recent' },
            forecast: { chartId: 'env-map-forecast' },
            turbidity: { selectId: 'env-metric-turbidity', chartId: 'env-map-turbidity' },
        };

        const ENV_FORECAST_WINDOWS = {
            '2040_2050': '2040–2050',
            '2090_2100': '2090–2100',
        };

        const ENV_FORECAST_STATS = {
            mean: 'ensemble mean',
            std: 'ensemble std dev',
            min: 'ensemble min',
            max: 'ensemble max',
        };

        const ENV_FORECAST_SCENARIOS = {
            ssp245: 'SSP2-4.5',
            ssp370: 'SSP3-7.0',
            ssp585: 'SSP5-8.5',
        };

        const ENV_COLORSCALES = {
            historical: 'RdYlBu_r',
            recent: 'YlOrRd',
            forecast: 'RdYlBu_r',
            turbidity: 'Viridis',
            cyclone: 'Hot',
            reefcheck: 'Viridis',
        };

        function methodsEnvPlotLayout(extraLayout = {}) {
            return {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                margin: { t: 24, r: 16, b: 24, l: 16 },
                ...extraLayout,
            };
        }

        function methodsEnvGeoLayout(colorbarTitle, { logScale = false, colorscale = 'Viridis' } = {}) {
            const coloraxis = {
                colorscale,
                colorbar: {
                    orientation: 'h',
                    title: { text: colorbarTitle, font: { size: 11 }, side: 'bottom' },
                    tickfont: { size: 10 },
                    len: 0.72,
                    thickness: 14,
                    y: -0.06,
                    yanchor: 'top',
                    x: 0.5,
                    xanchor: 'center',
                },
            };
            if (logScale) {
                coloraxis.type = 'log';
            }
            return methodsEnvPlotLayout({
                margin: { t: 24, r: 16, b: 72, l: 16 },
                geo: {
                    bgcolor: 'rgba(0,0,0,0)',
                    lakecolor: '#0f172a',
                    landcolor: '#1e293b',
                    subunitcolor: '#334155',
                    countrycolor: '#475569',
                    showland: true,
                    showcountries: true,
                    coastlinewidth: 0.4,
                    projection: { type: 'natural earth' },
                    domain: { x: [0, 1], y: [0.08, 1] },
                },
                coloraxis,
            });
        }

        function getEnvForecastScenario() {
            const select = document.getElementById('env-scenario-forecast');
            return select?.value || 'ssp370';
        }

        function getEnvForecastMetricKey() {
            const scenario = getEnvForecastScenario();
            const window = document.getElementById('env-window-forecast')?.value || '2040_2050';
            const stat = document.getElementById('env-stat-forecast')?.value || 'mean';
            return `sst_mean_${window}__${scenario}__model_${stat}`;
        }

        function getEnvForecastMetricLabel() {
            const scenario = getEnvForecastScenario();
            const window = document.getElementById('env-window-forecast')?.value || '2040_2050';
            const stat = document.getElementById('env-stat-forecast')?.value || 'mean';
            const windowLabel = ENV_FORECAST_WINDOWS[window] || window;
            const statLabel = ENV_FORECAST_STATS[stat] || stat;
            const scenarioLabel = ENV_FORECAST_SCENARIOS[scenario] || scenario;
            return `Mean SST (${windowLabel}), ${scenarioLabel} — ${statLabel}`;
        }

        function getEnvQdmExample() {
            const qdmRoot = methodsEnvData?.qdm;
            if (!qdmRoot) return null;
            if (qdmRoot.examples) {
                const select = document.getElementById('env-qdm-location');
                const locId = select?.value || qdmRoot.metadata?.default_loc_id;
                return qdmRoot.examples[locId] || Object.values(qdmRoot.examples)[0] || null;
            }
            if (qdmRoot.annual) return qdmRoot;
            return null;
        }

        function populateQdmExampleSelect() {
            const select = document.getElementById('env-qdm-location');
            const qdmRoot = methodsEnvData?.qdm;
            if (!select || !qdmRoot?.example_list?.length) return;

            const current = select.value;
            select.innerHTML = qdmRoot.example_list.map((ex) =>
                `<option value="${ex.loc_id}">${ex.label}</option>`
            ).join('');
            const defaultId = qdmRoot.metadata?.default_loc_id;
            const hasCurrent = current && qdmRoot.example_list.some((ex) => ex.loc_id === current);
            select.value = hasCurrent ? current : (defaultId || qdmRoot.example_list[0].loc_id);
        }

        function getEnvMetricsByGroup(group) {
            const meta = methodsEnvData?.locations?.metadata?.metrics || {};
            const entries = Object.entries(meta).filter(([, info]) => info.group === group);
            const sorted = entries.map(([key, info]) => ({ key, ...info }));
            sorted.sort((a, b) => (a.label || a.key).localeCompare(b.label || b.key));
            return sorted;
        }

        function populateEnvMetricSelects() {
            Object.entries(ENV_MAP_TARGETS).forEach(([group, { selectId }]) => {
                if (!selectId) return;
                const select = document.getElementById(selectId);
                if (!select) return;
                const metrics = getEnvMetricsByGroup(group);
                if (!metrics.length) return;

                const current = select.value;
                select.innerHTML = metrics.map((m) =>
                    `<option value="${m.key}">${m.label}</option>`
                ).join('');
                const hasCurrent = current && metrics.some((m) => m.key === current);
                select.value = hasCurrent ? current : metrics[0].key;
            });
        }

        function renderEnvMetricMap(group) {
            const target = ENV_MAP_TARGETS[group];
            if (!target || !methodsEnvData?.locations) return;

            const chartEl = document.getElementById(target.chartId);
            if (!chartEl) return;

            let metricKey;
            let metricMeta;
            if (group === 'forecast') {
                metricKey = getEnvForecastMetricKey();
                metricMeta = methodsEnvData.locations.metadata?.metrics?.[metricKey] || {
                    label: getEnvForecastMetricLabel(),
                    unit: '°C',
                };
            } else {
                const select = document.getElementById(target.selectId);
                if (!select) return;
                metricKey = select.value || getEnvMetricsByGroup(group)[0]?.key;
                if (!metricKey) return;
                metricMeta = methodsEnvData.locations.metadata?.metrics?.[metricKey] || {};
            }
            const { lat, lon, metrics } = methodsEnvData.locations;
            const values = metrics[metricKey] || [];
            const filtered = lat.map((la, i) => ({
                lat: la,
                lon: lon[i],
                value: values[i],
            })).filter((row) => row.lat != null && row.lon != null && Number.isFinite(row.value));

            const useLog = group === 'turbidity' || metricMeta.log_scale === 'true';
            const colorscale = ENV_COLORSCALES[group] || 'Viridis';
            const colorValues = filtered.map((r) => (
                useLog ? Math.max(r.value, 1e-4) : r.value
            ));

            const plotFn = chartEl.data ? Plotly.react : Plotly.newPlot;
            plotFn(target.chartId, [{
                type: 'scattergeo',
                lat: filtered.map((r) => r.lat),
                lon: filtered.map((r) => r.lon),
                marker: {
                    size: 5,
                    opacity: 0.75,
                    color: colorValues,
                    coloraxis: 'coloraxis',
                },
                hovertemplate:
                    `lat: %{lat:.2f}°<br>lon: %{lon:.2f}°<br>${metricMeta.label || metricKey}: %{customdata:.4f}${metricMeta.unit ? ` ${metricMeta.unit}` : ''}<extra></extra>`,
                customdata: filtered.map((r) => r.value),
            }], methodsEnvGeoLayout(metricMeta.label || metricKey, { logScale: useLog, colorscale }), { responsive: true });
        }

        function renderEnvQdmCharts() {
            const qdm = getEnvQdmExample();
            if (!qdm?.annual) return;

            const meta = qdm.metadata || {};
            const subtitle = document.getElementById('env-qdm-subtitle');
            if (subtitle) {
                subtitle.innerHTML =
                    `Example reef <strong>${meta.loc_id || ''}</strong> ` +
                    `(${meta.lat?.toFixed?.(2) ?? meta.lat}°, ${meta.lon?.toFixed?.(2) ?? meta.lon}°). ` +
                    `Raw historic → QDM forecast jump: <strong>${meta.jump_raw_to_qdm_forecast_c ?? '—'} °C</strong>; ` +
                    `after quantile mapping: <strong>${meta.jump_mapped_to_forecast_c ?? '—'} °C</strong>; ` +
                    `stitched series: <strong>${meta.jump_continuous_c ?? '—'} °C</strong>.`;
            }

            const refShape = [{
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: 1985,
                x1: 2014,
                y0: 0,
                y1: 1,
                fillcolor: 'rgba(148, 163, 184, 0.12)',
                line: { width: 0 },
                layer: 'below',
            }, {
                type: 'line',
                xref: 'x',
                yref: 'paper',
                x0: 2015,
                x1: 2015,
                y0: 0,
                y1: 1,
                line: { color: '#64748b', width: 1, dash: 'dash' },
            }];

            const annualEl = document.getElementById('env-qdm-annual');
            if (annualEl) {
                const plotFn = annualEl.data ? Plotly.react : Plotly.newPlot;
                plotFn('env-qdm-annual', [
                    {
                        x: qdm.annual.year,
                        y: qdm.annual.raw_cmip_c,
                        name: 'Raw CMIP historic',
                        mode: 'lines',
                        opacity: 0.45,
                        line: { color: '#3b82f6', width: 0.8, dash: 'dot' },
                        connectgaps: false,
                    },
                    {
                        x: qdm.annual.year,
                        y: qdm.annual.qm_mapped_historic_c,
                        name: 'QM-mapped historic',
                        mode: 'lines',
                        line: { color: '#E3B710', width: 1.2 },
                        connectgaps: false,
                    },
                    {
                        x: qdm.annual.year,
                        y: qdm.annual.qdm_historic_c,
                        name: 'QDM historic',
                        mode: 'lines',
                        line: { color: '#94a3b8', width: 1, dash: 'dash' },
                        connectgaps: false,
                    },
                    {
                        x: qdm.annual.year,
                        y: qdm.annual.qdm_forecast_c,
                        name: `QDM forecast (${meta.scenario || 'ssp370'})`,
                        mode: 'lines',
                        line: { color: '#F11B00', width: 1.2 },
                        connectgaps: false,
                    },
                    {
                        x: qdm.annual.year,
                        y: qdm.annual.continuous_c,
                        name: 'Stitched continuous',
                        mode: 'lines',
                        line: { color: valueTypeColor('tourism'), width: 1.6 },
                        connectgaps: false,
                    },
                ], methodsEnvPlotLayout({
                    height: 340,
                    xaxis: { gridcolor: '#334155', title: 'Year', range: [1978, 2057] },
                    yaxis: { gridcolor: '#334155', title: 'SST (°C)' },
                    legend: { orientation: 'h', y: -0.28, font: { size: 10 } },
                    margin: { t: 16, r: 16, b: 72, l: 52 },
                    shapes: refShape,
                    annotations: [{
                        x: 1999.5,
                        y: 1.03,
                        yref: 'paper',
                        text: 'QDM calibration',
                        showarrow: false,
                        font: { size: 10, color: '#94a3b8' },
                    }],
                }), { responsive: true });
            }

            const boundaryEl = document.getElementById('env-qdm-boundary');
            if (boundaryEl && qdm.boundary_zoom) {
                const plotFn = boundaryEl.data ? Plotly.react : Plotly.newPlot;
                plotFn('env-qdm-boundary', [
                    {
                        x: qdm.boundary_zoom.time,
                        y: qdm.boundary_zoom.raw_cmip_c,
                        name: 'Raw CMIP',
                        mode: 'lines',
                        opacity: 0.45,
                        line: { color: '#3b82f6', width: 0.8, dash: 'dot' },
                        connectgaps: false,
                    },
                    {
                        x: qdm.boundary_zoom.time,
                        y: qdm.boundary_zoom.qm_mapped_historic_c,
                        name: 'QM-mapped historic',
                        mode: 'lines',
                        line: { color: '#E3B710', width: 1.2 },
                        connectgaps: false,
                    },
                    {
                        x: qdm.boundary_zoom.time,
                        y: qdm.boundary_zoom.qdm_historic_c,
                        name: 'QDM historic',
                        mode: 'lines',
                        line: { color: '#94a3b8', width: 1, dash: 'dash' },
                        connectgaps: false,
                    },
                    {
                        x: qdm.boundary_zoom.time,
                        y: qdm.boundary_zoom.qdm_forecast_c,
                        name: 'QDM forecast',
                        mode: 'lines',
                        line: { color: '#F11B00', width: 1.2 },
                        connectgaps: false,
                    },
                ], methodsEnvPlotLayout({
                    height: 340,
                    xaxis: { gridcolor: '#334155', title: 'Month' },
                    yaxis: { gridcolor: '#334155', title: 'SST (°C)' },
                    legend: { orientation: 'h', y: -0.28, font: { size: 10 } },
                    margin: { t: 16, r: 16, b: 72, l: 52 },
                    shapes: [{
                        type: 'rect',
                        xref: 'x',
                        yref: 'paper',
                        x0: '2010-01',
                        x1: '2014-12',
                        y0: 0,
                        y1: 1,
                        fillcolor: 'rgba(148, 163, 184, 0.12)',
                        line: { width: 0 },
                        layer: 'below',
                    }, {
                        type: 'line',
                        x0: '2015-01',
                        x1: '2015-01',
                        y0: 0,
                        y1: 1,
                        yref: 'paper',
                        line: { color: '#64748b', width: 1, dash: 'dash' },
                    }],
                }), { responsive: true });
            }
        }

        function renderEnvCycloneChart() {
            const cyclone = methodsEnvData?.cyclone;
            const locations = methodsEnvData?.locations;
            const chartEl = document.getElementById('env-cyclone-chart');
            if (!cyclone || !chartEl) return;

            const traces = [{
                type: 'scattergeo',
                lat: cyclone.lat,
                lon: cyclone.lon,
                marker: {
                    size: 3,
                    opacity: 0.35,
                    color: cyclone.value,
                    colorscale: ENV_COLORSCALES.cyclone,
                    coloraxis: 'coloraxis',
                    cmin: 0,
                },
                name: 'Cyclone frequency grid',
                hovertemplate: 'lat: %{lat:.1f}°<br>lon: %{lon:.1f}°<br>freq: %{marker.color:.3f}<extra></extra>',
            }];

            if (locations?.lat?.length) {
                const reefVals = locations.metrics?.cyclone_freq || [];
                traces.push({
                    type: 'scattergeo',
                    lat: locations.lat,
                    lon: locations.lon,
                    marker: {
                        size: 4,
                        color: '#f8fafc',
                        line: { color: '#0f172a', width: 0.5 },
                        opacity: 0.85,
                    },
                    name: 'Reef locations',
                    hovertemplate:
                        'Reef<br>lat: %{lat:.2f}°<br>lon: %{lon:.2f}°<br>freq: %{customdata:.3f}<extra></extra>',
                    customdata: reefVals,
                });
            }

            // Move legend below the chart (orientation: horizontal, y below 0)
            const layout = methodsEnvGeoLayout('storms / cell / yr', { colorscale: ENV_COLORSCALES.cyclone });
            layout.legend = layout.legend || {};
            layout.legend.orientation = 'h';
            layout.legend.y = -0.2; // set y below the chart to place legend underneath
            layout.legend.x = 0.5;
            layout.legend.xanchor = 'center';

            const plotFn = chartEl.data ? Plotly.react : Plotly.newPlot;
            plotFn('env-cyclone-chart', traces, layout, { responsive: true });
        }

        function renderEnvReefcheckCharts() {
            const rc = methodsEnvData?.reefcheck;
            if (!rc) return;

            const yearEl = document.getElementById('env-reefcheck-year-chart');
            if (yearEl && rc.by_year?.length) {
                const years = rc.by_year.map((r) => r.year);
                const plotFn = yearEl.data ? Plotly.react : Plotly.newPlot;
                plotFn('env-reefcheck-year-chart', [
                    {
                        x: years,
                        y: rc.by_year.map((r) => r.n_surveys),
                        type: 'bar',
                        name: 'Surveys',
                        marker: { color: valueTypeColor('tourism') },
                    },
                    {
                        x: years,
                        y: rc.by_year.map((r) => r.n_sites),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Unique sites',
                        yaxis: 'y2',
                        line: { color: '#eab308', width: 2 },
                        marker: { size: 5 },
                    },
                ], methodsEnvPlotLayout({
                    barmode: 'overlay',
                    xaxis: { gridcolor: '#334155', title: 'Year', dtick: 5 },
                    yaxis: { gridcolor: '#334155', title: 'Surveys', rangemode: 'tozero' },
                    yaxis2: {
                        title: 'Unique sites',
                        overlaying: 'y',
                        side: 'right',
                        gridcolor: 'rgba(51,65,85,0.3)',
                        rangemode: 'tozero',
                    },
                    legend: { orientation: 'h', y: -0.18 },
                    margin: { t: 16, r: 55, b: 50, l: 50 },
                }), { responsive: true });
            }

            const mapEl = document.getElementById('env-reefcheck-map');
            if (mapEl && rc.points?.lat?.length) {
                const plotFn = mapEl.data ? Plotly.react : Plotly.newPlot;
                plotFn('env-reefcheck-map', [{
                    type: 'scattergeo',
                    lat: rc.points.lat,
                    lon: rc.points.lon,
                    marker: {
                        size: 3,
                        color: rc.points.year,
                        colorscale: ENV_COLORSCALES.reefcheck,
                        coloraxis: 'coloraxis',
                        opacity: 0.55,
                        cmin: rc.metadata?.year_min,
                        cmax: rc.metadata?.year_max,
                    },
                    hovertemplate: 'lat: %{lat:.2f}°<br>lon: %{lon:.2f}°<br>year: %{marker.color}<extra></extra>',
                }], methodsEnvGeoLayout('Survey year', { colorscale: ENV_COLORSCALES.reefcheck }), { responsive: true });
            }
        }

        function renderCotwDiversityMap() {
            const root = methodsEnvData?.cotwEcoregions;
            const chartEl = document.getElementById('env-cotw-diversity-map');
            if (!root?.geojson?.features?.length || !chartEl) return;

            const meta = root.metadata || {};
            document.querySelectorAll('.env-cotw-match-count').forEach((el) => {
                el.textContent = String(meta.n_with_diversity ?? '—');
            });
            document.querySelectorAll('.env-cotw-polygon-count').forEach((el) => {
                el.textContent = String(meta.n_polygons ?? '—');
            });

            const metric = document.getElementById('env-cotw-diversity-metric')?.value || 'diversity_standardized';
            const metricLabels = {
                diversity_standardized: 'Standardized (z-score) diversity (species count)',
                total_species_number: 'Total species count',
            };
            const features = root.geojson.features;
            const locations = [];
            const values = [];
            const hoverData = [];

            features.forEach((feature) => {
                const props = feature.properties || {};
                locations.push(String(props.erg));
                const raw = props[metric];
                values.push(raw != null && Number.isFinite(raw) ? raw : null);
                hoverData.push([
                    props.ecoregion || props.cotw_ecoregion_name || props.erg,
                    props.total_species_number,
                    props.diversity_standardized,
                ]);
            });

            const useLog = metric === 'total_species_number';
            const colorValues = values.map((value) => (
                useLog && value != null ? Math.max(value, 1) : value
            ));

            const plotFn = chartEl.data ? Plotly.react : Plotly.newPlot;
            plotFn('env-cotw-diversity-map', [{
                type: 'choropleth',
                geojson: root.geojson,
                featureidkey: 'properties.erg',
                locations,
                z: colorValues,
                marker: {
                    line: { color: '#475569', width: 0.35 },
                },
                hovertemplate:
                    '<b>%{customdata[0]}</b><br>' +
                    'Species: %{customdata[1]}<br>' +
                    'Std diversity: %{customdata[2]:.3f}<extra></extra>',
                customdata: hoverData,
                coloraxis: 'coloraxis',
            }], methodsEnvGeoLayout(metricLabels[metric] || metric, {
                logScale: useLog,
                colorscale: 'Viridis',
            }), { responsive: true });
        }

        function bindEnvMetricControls() {
            document.querySelectorAll('.env-metric-select').forEach((select) => {
                if (select.dataset.bound === '1') return;
                select.dataset.bound = '1';
                select.addEventListener('change', () => {
                    const group = select.dataset.envGroup;
                    if (group) renderEnvMetricMap(group);
                });
            });

            const scenarioSelect = document.getElementById('env-scenario-forecast');
            if (scenarioSelect && scenarioSelect.dataset.bound !== '1') {
                scenarioSelect.dataset.bound = '1';
                scenarioSelect.addEventListener('change', () => renderEnvMetricMap('forecast'));
            }

            document.querySelectorAll('.env-forecast-control').forEach((select) => {
                if (select.id === 'env-scenario-forecast' || select.dataset.bound === '1') return;
                select.dataset.bound = '1';
                select.addEventListener('change', () => renderEnvMetricMap('forecast'));
            });

            const qdmSelect = document.getElementById('env-qdm-location');
            if (qdmSelect && qdmSelect.dataset.bound !== '1') {
                qdmSelect.dataset.bound = '1';
                qdmSelect.addEventListener('change', () => renderEnvQdmCharts());
            }

            const cotwSelect = document.getElementById('env-cotw-diversity-metric');
            if (cotwSelect && cotwSelect.dataset.bound !== '1') {
                cotwSelect.dataset.bound = '1';
                cotwSelect.addEventListener('change', () => renderCotwDiversityMap());
            }
        }

        function renderMethodsEnvVisualizations() {
            if (!methodsEnvData) return;

            const countEls = document.querySelectorAll('.env-locations-count');
            if (countEls.length && methodsEnvData?.locations) {
                const n = String(methodsEnvData.locations.metadata?.n_locations ?? '—');
                countEls.forEach((el) => { el.textContent = n; });
            }

            if (methodsEnvData?.locations) {
                populateEnvMetricSelects();
                Object.keys(ENV_MAP_TARGETS).forEach(renderEnvMetricMap);
                populateQdmExampleSelect();
                renderEnvQdmCharts();
                renderEnvCycloneChart();
            }
            renderEnvReefcheckCharts();
            renderCotwDiversityMap();
            bindEnvMetricControls();
        }
        
        // ============================================================
        // MAP FUNCTIONS
        // ============================================================
        
        function initializeMap() {
            // Use canvas renderer for better performance with many polygons
            map = L.map('map', {
                preferCanvas: true,  // Use canvas renderer instead of SVG for better performance
                zoomControl: true
            }).setView([0, 0], 2);
            
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap, &copy; CARTO',
                maxZoom: 18
            }).addTo(map);

            // Pane ordering (Leaflet tilePane = 200; popupPane = 700):
            //   siteBackPane    250 – fisheries/coastal grid cells
            //   choroplethPane  380 – country choropleth
            //   sitePointPane   500 – legacy point markers
            //   sitePolygonPane 550 – tourism reef polygons (top map geometry;
            //                         keep below Leaflet popup/tooltip panes)
            map.createPane('siteBackPane');
            map.getPane('siteBackPane').style.zIndex = 250;
            map.createPane('choroplethPane');
            map.getPane('choroplethPane').style.zIndex = 380;
            map.createPane('sitePointPane');
            map.getPane('sitePointPane').style.zIndex = 500;
            map.createPane('sitePolygonPane');
            map.getPane('sitePolygonPane').style.zIndex = 550;

            // Dedicated renderers pinned to their panes so canvas batching cannot
            // paint grid cells above tourism polygons.
            siteGridRenderer = L.canvas({ pane: 'siteBackPane', padding: 0.5 });
            siteTourismRenderer = L.svg({ pane: 'sitePolygonPane' });
            
            choroplethLayer = L.layerGroup();
            siteGridLayer = L.layerGroup().addTo(map);
            siteTourismLayer = L.layerGroup().addTo(map);
            siteLayer = L.layerGroup().addTo(map);
            
            // Initialize legend with default settings (annual loss %, non-cumulative)
            const defaultScale = getColorScale(false, 'loss_percent');
            updateMapLegend(defaultScale, false);
            
            // Re-render on zoom/move for viewport-based rendering
            let zoomMoveTimeout = null;
            map.on('zoomend moveend', () => {
                if (!isRendering) {
                    clearTimeout(zoomMoveTimeout);
                    zoomMoveTimeout = setTimeout(() => {
                        // Reload chunked GeoJSON on viewport changes.
                        // In vector tile mode, tile loading is handled natively by Leaflet.VectorGrid.
                        if (!useVectorTilesForSites) {
                            loadSiteData();
                        }
                    }, 300);  // Debounce zoom/move events
                }
            });
            
            // Load country boundaries (using Natural Earth via CDN)
            loadCountryBoundaries();
        }
        
        async function loadCountryBoundaries() {
            try {
                // Prefer high-resolution country boundaries.
                const boundarySources = [
                    'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson',
                    'https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson',
                ];
                for (const url of boundarySources) {
                    try {
                        const response = await fetch(url);
                        if (response.ok) {
                            countryBoundaries = await response.json();
                            console.log(`Loaded country boundaries from: ${url}`);
                            return;
                        }
                    } catch (err) {
                        console.warn(`Boundary fetch failed for ${url}:`, err);
                    }
                }
                console.warn('Could not load country boundaries, choropleth will be unavailable');
            } catch (error) {
                console.warn('Error loading country boundaries:', error);
            }
        }

        function clearSiteDataLayers() {
            if (siteGridLayer) siteGridLayer.clearLayers();
            if (siteTourismLayer) siteTourismLayer.clearLayers();
            if (siteLayer) siteLayer.clearLayers();
        }

        function bringTourismLayersToFront() {
            if (siteTourismLayer) {
                siteTourismLayer.bringToFront();
            }
        }
        
        async function loadSiteData() {
            // Wait for siteManifest to be loaded
            if (!siteManifest) {
                console.warn('siteManifest not loaded yet, waiting...');
                // Try again after a short delay
                setTimeout(loadSiteData, 500);
                return;
            }
            
            if (!map || !siteLayer) {
                console.warn('Map not initialized yet');
                return;
            }
            
            // Store current view before loading new data (if not initial load)
            let currentView = null;
            if (!isInitialMapLoad && map) {
                currentView = {
                    center: map.getCenter(),
                    zoom: map.getZoom()
                };
            }
            
            const scenario = document.getElementById('map-scenario').value;
            const model = mapModelToCountryModel(document.getElementById('map-model').value);
            const selectedValueTypes = getMapSelectedValueTypes();
            
            if (!scenario || !model) {
                console.warn('Scenario or model not selected');
                return;
            }
            
            // Check if this is a cumulative scenario
            const isCumulative = scenario.startsWith('cumulative_');
            
            // For cumulative, scenario is like "cumulative_rcp45_2050", need to extract rcp and year
            let scenarioKey = scenario;
            if (isCumulative) {
                // Extract rcp and year from "cumulative_rcp45_2050"
                const parts = scenario.replace('cumulative_', '').split('_');
                const rcp = parts[0]; // "rcp45"
                const year = parts[1]; // "2050"
                scenarioKey = `cumulative_${rcp}_${year}`;
            }
            
            const metric = document.getElementById('map-metric').value;
            const scale = getColorScale(isCumulative, metric);
            const showChoropleth = document.getElementById('map-choropleth-toggle').checked;
            if (selectedValueTypes.length === 0) {
                clearSiteDataLayers();
                currentSiteGeojson = null;
                setMapEmptyState('Select at least one dataset to view map values.');
                updateMapLegend(scale, showChoropleth, 'No datasets selected');
                if (choroplethLayer) {
                    map.removeLayer(choroplethLayer);
                    choroplethLayer = L.layerGroup();
                }
                removeChoroplethLegend();
                return;
            }
            setMapEmptyState('');

            // Resolve manifest scenario key (handles linear relpct vs legacy pp suffixes).
            const buildScenarioKey = (datasetKey) =>
                resolveScenarioDatasetKey(datasetKey, scenarioKey, model, isCumulative);
            const datasetsToLoad = selectedValueTypes;
            const scenarioDatasetKeys = datasetsToLoad.map(buildScenarioKey);

            // Load gridded and/or viewport-tiled point datasets (mixed per value_type).
            const griddedManifest = isCumulative
                ? (siteManifest?.gridded_sites_cumulative || {})
                : (siteManifest?.gridded_sites_annual || {});
            const datasetTileIndex = isCumulative
                ? (siteManifest?.site_dataset_tiles_cumulative || {})
                : (siteManifest?.site_dataset_tiles_annual || {});
            const datasetTileZoom = Number(
                isCumulative
                    ? siteManifest?.site_dataset_tile_zoom_cumulative
                    : siteManifest?.site_dataset_tile_zoom_annual
            );
            const hasAnyMapSource = datasetsToLoad.some((datasetKey) =>
                Boolean(griddedManifest[datasetKey] || datasetTileIndex?.[datasetKey])
            );

            if (hasAnyMapSource) {
                try {
                    console.log('Loading site map data', {
                        scenario,
                        model,
                        scenarioDatasetKeys,
                        datasetsToLoad,
                    });
                    const {
                        features,
                        skippedPointDatasets,
                        minPointMapZoom,
                        canLoadPointsAtZoom,
                    } = await loadSelectedSiteFeatures({
                        datasetsToLoad,
                        scenarioKey,
                        model,
                        isCumulative,
                        buildScenarioKey,
                        griddedManifest,
                        datasetTileIndex,
                        datasetTileZoom,
                        mapBounds: map.getBounds(),
                        mapZoom: map.getZoom(),
                    });

                    if (features.length === 0) {
                        if (
                            skippedPointDatasets.length > 0 &&
                            skippedPointDatasets.length === datasetsToLoad.length
                        ) {
                            setMapEmptyState(
                                `Zoom in further to load coastal protection points.`
                            );
                            updateMapLegend(scale, showChoropleth, 'Point datasets load when zoomed in');
                        } else {
                            setMapEmptyState('No map data found for selected datasets.');
                            updateMapLegend(scale, showChoropleth, 'No map data for selection');
                        }
                        clearSiteDataLayers();
                        currentSiteGeojson = null;
                        isRendering = false;
                        return;
                    }

                    setMapEmptyState('');
                    console.log(`✓ Loaded ${features.length} map features`);
                    isRendering = false;
                    renderSites({ type: 'FeatureCollection', features }, isCumulative, currentView);
                    isInitialMapLoad = false;
                    return;
                } catch (error) {
                    console.error('Error loading site map data:', error);
                    setMapEmptyState('Error loading map data.');
                    isRendering = false;
                    return;
                }
            }

            const vectorTileIndex = siteManifest?.vector_tile_scenarios || {};
            const pointChunkIndex = siteManifest?.site_point_chunks || {};
            const chunkZoom = Number(siteManifest?.site_point_chunk_zoom);
            const effectiveTileZoom = Number.isFinite(datasetTileZoom) ? datasetTileZoom : chunkZoom;
            const visibleTileKeys = Number.isFinite(effectiveTileZoom)
                ? getTileKeysForBounds(map.getBounds(), effectiveTileZoom)
                : [];
            const minPointMapZoom = Number(siteManifest?.site_dataset_min_map_zoom ?? 4);
            const canLoadDatasetWidePointsAtZoom = map.getZoom() >= minPointMapZoom;
            const canUseDatasetWideTiles = Boolean(
                scenarioDatasetKeys.length > 0 &&
                canLoadDatasetWidePointsAtZoom &&
                datasetsToLoad.every((datasetKey) => {
                    const entry = datasetTileIndex?.[datasetKey];
                    return entry && entry.geometry && entry.attributes;
                })
            );

            const canUseVectorTiles = Boolean(
                L?.vectorGrid &&
                scenarioDatasetKeys.length > 0 &&
                !canUseDatasetWideTiles &&
                scenarioDatasetKeys.every((k) => Boolean(vectorTileIndex[k]?.url_template))
            );

            if (canUseVectorTiles) {
                useVectorTilesForSites = true;
                console.log('Loading site data using vector tiles', {
                    scenario,
                    model,
                    selectedValueTypes,
                    scenarioDatasetKeys,
                });
                renderVectorTileSites(
                    scenarioDatasetKeys,
                    vectorTileIndex,
                    isCumulative,
                    currentView
                );
                isInitialMapLoad = false;
                return;
            }

            useVectorTilesForSites = false;

            const filesToFetch = new Set();
            scenarioDatasetKeys.forEach((scenarioDatasetKey) => {
                filesToFetch.add(`sites_${scenarioDatasetKey}.json`);
                if (canUseDatasetWideTiles) {
                    return;
                }
                const tileMap = pointChunkIndex[scenarioDatasetKey];
                if (tileMap && visibleTileKeys.length > 0) {
                    visibleTileKeys.forEach((tileKey) => {
                        const chunkFile = tileMap[tileKey];
                        if (chunkFile) {
                            filesToFetch.add(chunkFile);
                        }
                    });
                }
            });
            const filenames = Array.from(filesToFetch);
            
            console.log('Loading site data:', {
                scenario,
                model,
                selectedValueTypes,
                isCumulative,
                scenarioKey,
                sanitizedModel,
                chunkZoom: effectiveTileZoom,
                mapZoom: map.getZoom(),
                minPointMapZoom,
                canUseDatasetWideTiles,
                requestedTiles: visibleTileKeys.length,
                filenames,
            });
            
            try {
                const [baseGeojsons, datasetWidePointFeatures] = await Promise.all([
                    Promise.all(
                        scenarioDatasetKeys.map((scenarioDatasetKey) =>
                            fetchGeoJsonWithCache(`sites_${scenarioDatasetKey}.json`)
                        )
                    ),
                    canUseDatasetWideTiles
                        ? Promise.all(
                            datasetsToLoad.map((datasetKey) =>
                                loadDatasetWidePointTileFeatures({
                                    datasetKey,
                                    scenarioDatasetKey: buildScenarioKey(datasetKey),
                                    tileKeys: visibleTileKeys,
                                    datasetTileIndex,
                                })
                            )
                        )
                        : Promise.resolve([]),
                ]);
                const validBaseGeojsons = baseGeojsons.filter(g => g && Array.isArray(g.features));
                let features = validBaseGeojsons.flatMap((g) => g.features || []);

                if (canUseDatasetWideTiles) {
                    features = features.concat(datasetWidePointFeatures.flat());
                } else {
                    const extraGeojsons = await Promise.all(
                        filenames
                            .filter((name) => !name.startsWith('sites_'))
                            .map((filename) => fetchGeoJsonWithCache(filename))
                    );
                    const validExtraGeojsons = extraGeojsons.filter(g => g && Array.isArray(g.features));
                    features = features.concat(validExtraGeojsons.flatMap((g) => g.features || []));
                }

                if (features.length === 0) {
                    console.error('No site datasets found for selection', { filenames });
                    clearSiteDataLayers();
                    currentSiteGeojson = null;
                    if (!canLoadDatasetWidePointsAtZoom && datasetsToLoad.some((k) => Boolean(datasetTileIndex?.[k]))) {
                        setMapEmptyState(`Zoom in further to load point datasets.`);
                        updateMapLegend(scale, showChoropleth, 'Point datasets load when zoomed in');
                    } else {
                        setMapEmptyState('No map data found for selected datasets.');
                        updateMapLegend(scale, showChoropleth, 'No datasets with available map data');
                    }
                    return;
                }
                setMapEmptyState('');
                const geojson = {
                    type: 'FeatureCollection',
                    features,
                };
                
                if (!geojson || !geojson.features || geojson.features.length === 0) {
                    console.warn('GeoJSON has no features for current viewport');
                    clearSiteDataLayers();
                    currentSiteGeojson = null;
                    setMapEmptyState('No features available in the current map view.');
                    updateMapLegend(scale, showChoropleth, 'No features in current viewport');
                    return;
                }
                
                console.log(`✓ Loaded ${geojson.features.length} features from ${filenames.length} file(s)`);
                renderSites(geojson, isCumulative, currentView);
                
                // Mark that initial load is complete
                isInitialMapLoad = false;
            } catch (error) {
                console.error('Error loading site data:', error);
                console.error('Error details:', error.message, error.stack);
                clearSiteDataLayers();
                currentSiteGeojson = null;
            }
        }
        
        // Four-bin palette tuned for dark basemaps (green → lime → orange → red).
        const MAP_LOSS_COLORS = ['#2d9f4f', '#b8d62e', '#ff8c42', '#e01a4f'];

        // Eleven-bin baseline palette (matches notebooks/economic_datasets.ipynb prices_df).
        const MAP_BASELINE_VALUE_COLORS = [
            '#828282', '#2892c8', '#74b474', '#52bf81', '#59d95e',
            '#58f230', '#f6e058', '#e6ac3e', '#d57726', '#bf4713', '#730000',
        ];
        const MAP_BASELINE_VALUE_BREAKS = [
            4000, 8000, 12000, 24000, 44000, 92000, 172000, 352000, 908000,
        ];
        const MAP_BASELINE_VALUE_LABELS = [
            'no value',
            '≤4',
            '4–8',
            '8–12',
            '12–24',
            '24–44',
            '44–92',
            '92–172',
            '172–352',
            '352–908',
            '>908',
        ];
        const MAP_BASELINE_VALUE_SCALE = {
            breaks: MAP_BASELINE_VALUE_BREAKS,
            colors: MAP_BASELINE_VALUE_COLORS,
            labels: MAP_BASELINE_VALUE_LABELS,
            title: 'Baseline economic value (thousands USD)',
            subtitle: 'Fixed log-spaced bins',
            binMode: 'fixedUpperBounds',
        };

        const COLOR_SCALE_DEFAULTS = {
            loss_percent_annual: {
                breaks: [0.05, 0.15, 0.35],
                labels: ['< 5%', '5–15%', '15–35%', '≥ 35%'],
                title: 'Annual loss %',
                colors: MAP_LOSS_COLORS,
                subtitle: 'Quartile bins (visible extent)',
            },
            loss_percent_cumulative: {
                breaks: [0.05, 0.15, 0.35],
                labels: ['< 5%', '5–15%', '15–35%', '≥ 35%'],
                title: 'Annual loss % (endpoint year)',
                colors: MAP_LOSS_COLORS,
                subtitle: 'Quartile bins (visible extent)',
            },
            absolute_annual: {
                breaks: [50000, 500000, 5000000],
                labels: ['< $50k', '$50k–$500k', '$500k–$5M', '≥ $5M'],
                title: 'Projected annual loss (USD)',
                colors: MAP_LOSS_COLORS,
                subtitle: 'Log-scaled quartile bins (visible extent); loss from coral decline',
            },
            absolute_cumulative: {
                breaks: [500000, 5000000, 50000000],
                labels: ['< $500k', '$500k–$5M', '$5M–$50M', '≥ $50M'],
                title: 'Projected cumulative loss (USD)',
                colors: MAP_LOSS_COLORS,
                subtitle: 'Log-scaled quartile bins (visible extent); loss from coral decline',
            },
            baseline_value: {
                ...MAP_BASELINE_VALUE_SCALE,
            },
        };

        function getScaleKey(isCumulative, metric) {
            if (metric === 'baseline_value') {
                return 'baseline_value';
            }
            if (metric === 'loss_percent') {
                return isCumulative ? 'loss_percent_cumulative' : 'loss_percent_annual';
            }
            return isCumulative ? 'absolute_cumulative' : 'absolute_annual';
        }

        function resolveFeatureMetricValue(props, metric, isCumulative) {
            if (metric === 'baseline_value') {
                return Number(props.original_value || 0);
            }
            if (metric === 'loss_percent') {
                return props.loss_fraction || 0;
            }
            return isCumulative
                ? (props.cumulative_loss || 0)
                : (props.value_loss || props.annual_loss || 0);
        }

        function resolveCountryMetricValue(countryRow, metric, isCumulative) {
            if (metric === 'baseline_value') {
                return Number(countryRow.original_value || 0);
            }
            if (metric === 'loss_percent') {
                return countryRow.loss_fraction || 0;
            }
            return isCumulative
                ? (countryRow.cumulative_loss || 0)
                : (countryRow.value_loss || 0);
        }

        function getColorScale(isCumulative, metric) {
            const key = getScaleKey(isCumulative, metric);
            const defaults = COLOR_SCALE_DEFAULTS[key];
            if (metric === 'baseline_value') {
                return {
                    ...MAP_BASELINE_VALUE_SCALE,
                    metric,
                    min: 0,
                    max: MAP_BASELINE_VALUE_BREAKS[MAP_BASELINE_VALUE_BREAKS.length - 1],
                };
            }
            return {
                ...defaults,
                colors: defaults.colors || MAP_LOSS_COLORS,
                metric,
                min: 0,
                max: defaults.breaks[2],
            };
        }

        function quantile(sorted, p) {
            if (!sorted.length) return 0;
            const idx = (sorted.length - 1) * p;
            const lo = Math.floor(idx);
            const hi = Math.ceil(idx);
            if (lo === hi) return sorted[lo];
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
        }

        function formatMetricValue(value, metric) {
            if (value == null || !Number.isFinite(value)) return 'n/a';
            if (metric === 'loss_percent') {
                return `${(value * 100).toFixed(1)}%`;
            }
            const abs = Math.abs(value);
            if (abs >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
            if (abs >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
            if (abs >= 1e3) return `$${(value / 1e3).toFixed(0)}k`;
            return `$${Number(value).toFixed(0)}`;
        }

        function buildColorScaleFromValues(values, metric, isCumulative) {
            const defaults = getColorScale(isCumulative, metric);
            if (metric === 'baseline_value') {
                return {
                    ...MAP_BASELINE_VALUE_SCALE,
                    metric,
                    dataDriven: false,
                };
            }
            // Monetary layers are highly right-skewed, so log bins produce a more
            // interpretable discrete legend than linear spacing.
            const useLogScale = metric !== 'loss_percent';
            const clean = values.filter((v) => Number.isFinite(v) && (!useLogScale || v > 0));
            if (clean.length === 0) {
                return { ...defaults, dataDriven: false };
            }

            const scaleValues = useLogScale ? clean.map((v) => Math.log10(v)) : clean;
            const sorted = [...scaleValues].sort((a, b) => a - b);
            const sortedClean = [...clean].sort((a, b) => a - b);
            const min = sortedClean[0];
            const max = sortedClean[sortedClean.length - 1];

            if (sorted[0] === sorted[sorted.length - 1]) {
                return {
                    ...defaults,
                    breaks: [min, min, min],
                    labels: [
                        formatMetricValue(min, metric),
                        '—',
                        '—',
                        formatMetricValue(max, metric),
                    ],
                    min,
                    max,
                    dataDriven: true,
                };
            }

            let breaks = [
                quantile(sorted, 0.25),
                quantile(sorted, 0.5),
                quantile(sorted, 0.75),
            ];
            if (useLogScale) {
                breaks = breaks.map((b) => 10 ** b);
            }
            const span = Math.max((useLogScale ? max / min : max - min), Number.EPSILON);
            const epsilon = useLogScale ? span * 0.02 : span * 0.02;
            for (let i = 0; i < breaks.length; i += 1) {
                if (i > 0 && breaks[i] <= breaks[i - 1]) {
                    breaks[i] = breaks[i - 1] + epsilon;
                }
            }
            if (breaks[2] >= max) {
                breaks[2] = max - epsilon;
            }
            if (breaks[1] >= breaks[2]) {
                breaks[1] = (breaks[0] + breaks[2]) / 2;
            }

            const labels = [
                `< ${formatMetricValue(breaks[0], metric)}`,
                `${formatMetricValue(breaks[0], metric)} – ${formatMetricValue(breaks[1], metric)}`,
                `${formatMetricValue(breaks[1], metric)} – ${formatMetricValue(breaks[2], metric)}`,
                `≥ ${formatMetricValue(breaks[2], metric)}`,
            ];

            return {
                breaks,
                colors: defaults.colors || MAP_LOSS_COLORS,
                labels,
                title: defaults.title,
                subtitle: defaults.subtitle,
                metric,
                min,
                max,
                dataDriven: true,
            };
        }

        function getColorFromScale(value, scale) {
            if (scale.binMode === 'fixedUpperBounds') {
                if (value == null || !Number.isFinite(value) || value <= 0) {
                    return scale.colors[0];
                }
                for (let i = 0; i < scale.breaks.length; i += 1) {
                    if (value <= scale.breaks[i]) {
                        return scale.colors[i + 1];
                    }
                }
                return scale.colors[scale.colors.length - 1];
            }
            if (!Number.isFinite(value)) return scale.colors[0];
            if (value < scale.breaks[0]) return scale.colors[0];
            if (value < scale.breaks[1]) return scale.colors[1];
            if (value < scale.breaks[2]) return scale.colors[2];
            return scale.colors[3];
        }

        function setMapEmptyState(message = '') {
            const el = document.getElementById('map-empty-state');
            if (!el) return;
            if (message) {
                el.textContent = message;
                el.style.display = 'flex';
            } else {
                el.style.display = 'none';
            }
        }

        function updateMapLegend(scale, showChoropleth = false, statusMessage = '') {
            const legendEl = document.getElementById('map-legend');
            if (!legendEl) return;

            const subtitle = scale.subtitle
                || (scale.dataDriven ? 'Quartile bins (visible extent)' : 'Default bins (no visible data)');

            let html = `
                <div class="legend-scale-header">
                    <div class="legend-title">${scale.title}</div>
                    <div class="legend-subtitle">${subtitle}</div>
                </div>`;

            if (scale.binMode === 'fixedUpperBounds') {
                html += `
                    <div class="legend-discrete-bar">
                        <div class="legend-discrete-bar-swatches" style="grid-template-columns: repeat(${scale.colors.length}, 1fr);">
                            ${scale.colors.map((color) => `<div style="background:${color};"></div>`).join('')}
                        </div>
                        <div class="legend-discrete-bar-labels" style="grid-template-columns: repeat(${scale.colors.length}, 1fr);">
                            ${scale.labels.map((label) => `<span>${label}</span>`).join('')}
                        </div>
                    </div>`;
            } else {
                for (let i = 0; i < scale.colors.length; i += 1) {
                    html += `
                        <div class="legend-item">
                            <div class="legend-color" style="background: ${scale.colors[i]};"></div>
                            <span>${scale.labels[i]}</span>
                        </div>`;
                }
            }

            if (showChoropleth) {
                html += `
                    <div class="legend-separator"></div>
                    <div class="legend-item" id="choropleth-legend">
                        <span class="legend-note">Dashed fill = no country data</span>
                    </div>`;
            }

            if (statusMessage) {
                html += `
                    <div class="legend-item">
                        <span class="legend-note">${statusMessage}</span>
                    </div>`;
            }

            const mapModel = document.getElementById('map-model')?.value || '';
            if (String(mapModel).includes('Linear')) {
                html += `
                    <div class="legend-item">
                        <span class="legend-note">Fisheries linear: habitat α = ${getHabitatExportA()}</span>
                    </div>`;
            }

            legendEl.innerHTML = html;
        }
        
        // Simple geometry simplification function (only at very low zoom levels)
        function simplifyGeometry(geometry, tolerance = 0.0001) {
            // Only simplify if tolerance is very high (low zoom), otherwise return original
            if (tolerance < 0.0005) {
                return geometry;  // Don't simplify at normal zoom levels
            }
            
            if (!geometry || !geometry.coordinates) return geometry;
            
            function simplifyRing(ring, tol) {
                if (ring.length <= 2) return ring;
                const simplified = [ring[0]];
                for (let i = 1; i < ring.length - 1; i++) {
                    const prev = ring[i - 1];
                    const curr = ring[i];
                    const next = ring[i + 1];
                    // Simple distance check - keep point if it's far enough from line
                    const dx1 = curr[0] - prev[0];
                    const dy1 = curr[1] - prev[1];
                    const dx2 = next[0] - prev[0];
                    const dy2 = next[1] - prev[1];
                    const cross = Math.abs(dx1 * dy2 - dx2 * dy1);
                    if (cross > tol) {
                        simplified.push(curr);
                    }
                }
                simplified.push(ring[ring.length - 1]);
                return simplified;
            }
            
            if (geometry.type === 'Polygon') {
                return {
                    type: 'Polygon',
                    coordinates: geometry.coordinates.map(ring => simplifyRing(ring, tolerance))
                };
            } else if (geometry.type === 'MultiPolygon') {
                return {
                    type: 'MultiPolygon',
                    coordinates: geometry.coordinates.map(polygon => 
                        polygon.map(ring => simplifyRing(ring, tolerance))
                    )
                };
            }
            return geometry;
        }
        
        // Extra cells kept around the viewport so partially visible grid cells / polygons
        // are not dropped when panning or zooming.
        const VIEWPORT_CELL_BUFFER = 2;

        function resolveGridCellSizeDeg(geojson) {
            if (siteGridResolutionDeg) return siteGridResolutionDeg;
            for (const feature of geojson?.features || []) {
                const res = Number(feature?.properties?.grid_resolution_deg);
                if (Number.isFinite(res) && res > 0) return res;
            }
            const annual = siteManifest?.gridded_sites_annual || {};
            for (const entry of Object.values(annual)) {
                const res = Number(entry?.grid_resolution_deg);
                if (Number.isFinite(res) && res > 0) return res;
            }
            return null;
        }

        // Filter features by viewport bounds
        function filterFeaturesByViewport(geojson, mapBounds) {
            if (!mapBounds) return geojson;

            const cellSize = resolveGridCellSizeDeg(geojson);
            const pad = cellSize ? VIEWPORT_CELL_BUFFER * cellSize : 0;
            const west = mapBounds.getWest() - pad;
            const east = mapBounds.getEast() + pad;
            const south = mapBounds.getSouth() - pad;
            const north = mapBounds.getNorth() + pad;
            
            const filteredFeatures = geojson.features.filter(feature => {
                if (!feature.geometry || !feature.geometry.coordinates) return false;
                
                // Get bounding box of feature
                let minLon = Infinity, maxLon = -Infinity;
                let minLat = Infinity, maxLat = -Infinity;
                
                function processCoords(coords) {
                    if (Array.isArray(coords[0])) {
                        coords.forEach(processCoords);
                    } else {
                        const [lon, lat] = coords;
                        minLon = Math.min(minLon, lon);
                        maxLon = Math.max(maxLon, lon);
                        minLat = Math.min(minLat, lat);
                        maxLat = Math.max(maxLat, lat);
                    }
                }
                
                processCoords(feature.geometry.coordinates);
                
                return !(maxLon < west || minLon > east || maxLat < south || minLat > north);
            });
            
            return {
                type: 'FeatureCollection',
                features: filteredFeatures
            };
        }

        function renderVectorTileSites(scenarioDatasetKeys, vectorTileIndex, isCumulative = false, preserveView = null) {
            if (!map || !siteLayer) {
                return;
            }
            if (isRendering) {
                return;
            }
            isRendering = true;
            clearSiteDataLayers();
            currentSiteGeojson = null;

            const metric = document.getElementById('map-metric').value;
            const scale = getColorScale(isCumulative, metric);
            const showChoropleth = document.getElementById('map-choropleth-toggle').checked;
            const zoom = map.getZoom();
            const lossType = isCumulative ? 'Cumulative' : 'Annual';

            const formatMoney = (val) => {
                if (val >= 1e6) return `$${(val / 1e6).toFixed(2)}M`;
                if (val >= 1e3) return `$${(val / 1e3).toFixed(1)}k`;
                return `$${val.toFixed(0)}`;
            };

            const getFeatureColorValue = (props) =>
                resolveFeatureMetricValue(props || {}, metric, isCumulative);

            const styleByGeometry = (properties, z, geometryDimension) => {
                const colorValue = getFeatureColorValue(properties || {});
                const fillColor = getColorFromScale(colorValue, scale);
                // Use a geometry-agnostic style so points remain visible even if
                // geometryDimension semantics vary across vector tile sources.
                return {
                    radius: getPointRadius(z),
                    fill: true,
                    fillColor: fillColor,
                    fillOpacity: 0.75,
                    color: '#0f172a',
                    weight: 0.8,
                    opacity: 0.95,
                };
            };

            try {
                scenarioDatasetKeys.forEach((scenarioKey) => {
                    const entry = vectorTileIndex[scenarioKey];
                    if (!entry?.url_template) return;

                    const layerName = entry.layer || 'sites';
                    const tileUrl = DATA_PATH + entry.url_template;
                    const vectorLayer = L.vectorGrid.protobuf(tileUrl, {
                        interactive: true,
                        pane: 'sitePointPane',
                        maxNativeZoom: Number(entry.max_zoom || 14),
                        vectorTileLayerStyles: {
                            [layerName]: styleByGeometry
                        }
                    });

                    vectorLayer.on('click', (e) => {
                        const props = e.layer?.properties || {};
                        const lossFraction = Number(props.loss_fraction || 0);
                        const lossValue = isCumulative
                            ? Number(props.cumulative_loss || 0)
                            : Number(props.value_loss || props.annual_loss || 0);
                        const popupContent = `
                            <div class="popup-content">
                                <strong>${props.country || 'Unknown'}</strong><br>
                                Dataset: ${formatValueType(props.value_type || 'unknown')}<br>
                                <span style="color: #94a3b8;">${describeValueType(props.value_type)}</span><br>
                                Original Value (annual): ${formatMoney(Number(props.original_value || 0))}<br>
                                ${lossType} Loss: ${formatMoney(lossValue)} (${(lossFraction * 100).toFixed(1)}%)<br>
                                Coral change: ${(Number(props.coral_change || 0) * 100).toFixed(1)}pp
                            </div>
                        `;
                        L.popup().setLatLng(e.latlng).setContent(popupContent).openOn(map);
                    });

                    siteLayer.addLayer(vectorLayer);
                });

                updateMapLegend(scale, showChoropleth, 'Vector tiles mode (zoom/pan loaded natively)');

                if (!preserveView) {
                    map.setView(map.getCenter(), zoom);
                }
                if (showChoropleth) {
                    renderChoropleth();
                }
                bringTourismLayersToFront();
            } catch (error) {
                console.error('Error rendering vector tiles:', error);
            } finally {
                isRendering = false;
            }
        }
        
        function renderSites(geojson, isCumulative = false, preserveView = null) {
            if (!geojson || !geojson.features || geojson.features.length === 0) {
                console.warn('No features to render');
                clearSiteDataLayers();
                isRendering = false;
                return;
            }
            
            if (!map || !siteLayer) {
                console.error('Map or siteLayer not initialized');
                isRendering = false;
                return;
            }
            
            // Prevent concurrent renders
            if (isRendering) {
                return;
            }
            isRendering = true;
            
            clearSiteDataLayers();
            currentSiteGeojson = geojson;
            
            const metric = document.getElementById('map-metric').value;
            const showChoropleth = document.getElementById('map-choropleth-toggle').checked;
            
            // Get current zoom level
            const zoom = map.getZoom();
            
            // Filter features by viewport (only render visible features)
            const mapBounds = map.getBounds();
            const filteredGeoJSON = filterFeaturesByViewport(geojson, mapBounds);
            
            console.log(`Filtered ${geojson.features.length} features to ${filteredGeoJSON.features.length} visible features`);

            const isPointGeometry = (geometry) => {
                const type = geometry?.type;
                return type === 'Point' || type === 'MultiPoint';
            };

            const getFeatureColorValue = (props) =>
                resolveFeatureMetricValue(props || {}, metric, isCumulative);

            const visibleValues = filteredGeoJSON.features
                .map((feature) => getFeatureColorValue(feature.properties || {}))
                .filter((value) => Number.isFinite(value));
            const visibleValueTypes = [
                ...new Set(
                    filteredGeoJSON.features
                        .map((feature) => feature?.properties?.value_type)
                        .filter(Boolean)
                ),
            ];
            const scale = buildColorScaleFromValues(visibleValues, metric, isCumulative);
            currentMapColorScale = scale;
            const datasetMessage = visibleValueTypes.length
                ? `Showing: ${visibleValueTypes.map((vt) => formatValueType(vt)).join(', ')}`
                : 'No features in current viewport';

            updateMapLegend(scale, showChoropleth, datasetMessage);
            
            // Format values for popup
            const formatMoney = (val) => {
                if (val >= 1e6) return `$${(val / 1e6).toFixed(2)}M`;
                if (val >= 1e3) return `$${(val / 1e3).toFixed(1)}k`;
                return `$${val.toFixed(0)}`;
            };
            
            const lossType = isCumulative ? 'Cumulative' : 'Annual';
            
            // Create style function that will be applied to each feature
            const styleFunction = (feature) => {
                const props = feature.properties;
                
                const colorValue = resolveFeatureMetricValue(props, metric, isCumulative);
                
                const fillColor = getColorFromScale(colorValue, scale);
                
                return {
                    pane: 'sitePolygonPane',
                    renderer: siteTourismRenderer,
                    fillColor: fillColor,
                    color: '#1a2332',
                    weight: zoom < 5 ? 0.5 : 1,  // Thinner lines at low zoom
                    opacity: 0.9,
                    fillOpacity: 0.7
                };
            };

            const DATASET_FILL_OPACITY = { coastal_protection: 0.65, fisheries: 0.75, tourism: 0.88 };

            const pointToLayer = (feature, latlng) => {
                const props = feature.properties || {};
                const colorValue = getFeatureColorValue(props);
                const fillColor = getColorFromScale(colorValue, scale);
                const gridRes = Number(props.grid_resolution_deg);
                if (Number.isFinite(gridRes) && gridRes > 0) {
                    // Fisheries/coastal-protection only: exact-fit rectangles in siteBackPane.
                    const half = gridRes / 2;
                    const bounds = [
                        [latlng.lat - half, latlng.lng - half],
                        [latlng.lat + half, latlng.lng + half],
                    ];
                    return L.rectangle(bounds, {
                        pane: 'siteBackPane',
                        renderer: siteGridRenderer,
                        fillColor,
                        color: 'transparent',
                        weight: 0,
                        fillOpacity: DATASET_FILL_OPACITY[props.value_type] ?? 0.72,
                    });
                }
                const aligned = applyPointAlignmentOffset(latlng);
                return L.circleMarker(aligned, {
                    pane: 'sitePointPane',
                    radius: getPointRadius(zoom),
                    fillColor,
                    color: '#0f172a',
                    weight: 0.8,
                    opacity: 0.95,
                    fillOpacity: 0.8,
                    stroke: false
                });
            };
            
            // Create popup function
            const onEachFeature = (feature, layer) => {
                const props = feature.properties;
                const lossFraction = props.loss_fraction || 0;
                const lossValue = isCumulative ?
                    (props.cumulative_loss || 0) :
                    (props.value_loss || props.annual_loss || 0);
                
                // const layerLabel = props.n_sites > 1
                //     ? `Grid cell (${props.n_sites} sites)`
                //     : (isPointGeometry(feature.geometry) ? 'Point layer' : 'Polygon layer');
                const popupContent = `
                    <div class="popup-content">
                        <strong>${props.country || 'Unknown'}</strong><br>
                        Dataset: ${formatValueType(props.value_type || 'unknown')}<br>
                        <span style="color: #94a3b8;">${describeValueType(props.value_type)}</span><br>
                        Original Value (annual): ${formatMoney(props.original_value || 0)}<br>
                        ${lossType} Loss: ${formatMoney(lossValue)} (${(lossFraction * 100).toFixed(1)}%)<br>
                        Coral change: ${((props.coral_change || 0) * 100).toFixed(1)}pp
                    </div>
                `;
                layer.bindPopup(popupContent);
            };
            
            // Only simplify at very low zoom levels (zoom < 3) to preserve shape quality
            // Use filtered GeoJSON directly without simplification for normal zoom levels
            const simplifiedGeoJSON = zoom < 3 ? {
                type: 'FeatureCollection',
                features: filteredGeoJSON.features.map(feature => ({
                    ...feature,
                    geometry: simplifyGeometry(feature.geometry, 0.001)  // Only at very low zoom
                }))
            } : filteredGeoJSON;
            
            // Split grid points and tourism polygons into separate layer groups so
            // pane/renderer z-order is respected (canvas batching otherwise stacks
            // grid cells above reef polygons).
            const gridFeatures = simplifiedGeoJSON.features.filter((f) =>
                isPointGeometry(f.geometry)
            );
            const tourismFeatures = simplifiedGeoJSON.features.filter((f) =>
                !isPointGeometry(f.geometry)
            );

            // Render features using requestAnimationFrame for smooth rendering
            requestAnimationFrame(() => {
                try {
                    let layerCount = 0;

                    if (gridFeatures.length > 0) {
                        const gridGeoJson = L.geoJSON(
                            { type: 'FeatureCollection', features: gridFeatures },
                            { pointToLayer, onEachFeature }
                        );
                        gridGeoJson.eachLayer((layer) => {
                            siteGridLayer.addLayer(layer);
                            layerCount++;
                        });
                    }

                    if (tourismFeatures.length > 0) {
                        const tourismGeoJson = L.geoJSON(
                            { type: 'FeatureCollection', features: tourismFeatures },
                            { style: styleFunction, onEachFeature }
                        );
                        tourismGeoJson.eachLayer((layer) => {
                            siteTourismLayer.addLayer(layer);
                            layerCount++;
                        });
                    }

                    bringTourismLayersToFront();
                    
                    console.log(`✓ Added ${layerCount} layers to map (simplified, zoom: ${zoom})`);
                    
                    // Only fit bounds on initial load, otherwise preserve current view
                    if (layerCount > 0 && !preserveView) {
                        // Initial load: fit bounds to show all polygons
                        try {
                            const bounds = L.geoJSON(geojson).getBounds();
                            if (bounds && bounds.isValid()) {
                                map.fitBounds(bounds, { 
                                    padding: [50, 50],
                                    maxZoom: 10
                                });
                                console.log(`✓ Map bounds set to show all polygons`);
                            } else {
                                console.warn('Invalid bounds, using default view');
                                map.setView([0, 0], 2);
                            }
                        } catch (boundsError) {
                            console.warn('Error setting map bounds:', boundsError);
                            map.setView([0, 0], 2);
                        }
                    } else if (preserveView) {
                        const currentCenter = map.getCenter();
                        const centerChanged =
                            Math.abs(currentCenter.lat - preserveView.center.lat) > 1e-9 ||
                            Math.abs(currentCenter.lng - preserveView.center.lng) > 1e-9;
                        const zoomChanged = map.getZoom() !== preserveView.zoom;
                        if (centerChanged || zoomChanged) {
                            map.setView(preserveView.center, preserveView.zoom);
                        }
                    }
                    
                    // Update choropleth if it's enabled
                    if (showChoropleth) {
                        renderChoropleth();
                    }
                    // Ensure points stay on top after any redraw.
                    bringTourismLayersToFront();
                } catch (error) {
                    console.error('Error rendering sites:', error);
                } finally {
                    isRendering = false;
                }
            });
        }

        function toggleChoropleth() {
            const enabled = document.getElementById('map-choropleth-toggle').checked;
            const scenario = document.getElementById('map-scenario').value;
            const isCumulative = scenario.startsWith('cumulative_');
            const metric = document.getElementById('map-metric').value;
            const scale = currentMapColorScale || getColorScale(isCumulative, metric);
            
            if (enabled) {
                renderChoropleth();
            } else {
                if (choroplethLayer) {
                    map.removeLayer(choroplethLayer);
                    choroplethLayer = L.layerGroup();
                }
                updateMapLegend(scale, false, '');
                removeChoroplethLegend();
            }
        }
        
        function renderChoropleth() {
            const scenario = document.getElementById('map-scenario').value;
            const isCumulative = scenario.startsWith('cumulative_');
            const metric = document.getElementById('map-metric').value;
            const selectedValueTypes = getMapSelectedValueTypes();
            const dataSource = isCumulative ? cumulativeCountryData : countryData;
            if (selectedValueTypes.length === 0) {
                if (choroplethLayer) {
                    map.removeLayer(choroplethLayer);
                    choroplethLayer = L.layerGroup();
                }
                return;
            }
            
            if (!dataSource || !countryBoundaries) {
                console.warn('Country data or boundaries not loaded. dataSource:', !!dataSource, 'countryBoundaries:', !!countryBoundaries, 'isCumulative:', isCumulative);
                if (isCumulative) {
                    console.log('cumulativeCountryData:', cumulativeCountryData ? `loaded (${cumulativeCountryData.length} records)` : 'not loaded');
                } else {
                    console.log('countryData:', countryData ? `loaded (${countryData.length} records)` : 'not loaded');
                }
                return;
            }
            
            const model = mapModelToCountryModel(document.getElementById('map-model').value);

            console.log('Rendering choropleth for scenario:', scenario, 'model:', model, 'isCumulative:', isCumulative, 'metric:', metric);
            
            // Filter country data for current scenario and model
            const filtered = dataSource.filter(c => {
                const cScenario = c.scenario.toLowerCase();
                const targetScenario = scenario.toLowerCase();
                const scenarioMatch = cScenario === targetScenario;
                const modelMatch = c.model === model;
                const valueTypeMatch = selectedValueTypes.includes(c.value_type);
                return scenarioMatch && modelMatch && valueTypeMatch;
            });
            const filteredData = selectedValueTypes.length > 1
                ? aggregateRowsByCountry(filtered.map(r => ({ ...r, scenario, model })), isCumulative, false)
                : filtered;
            
            console.log('Filtered countries:', filteredData.length);

            let scale = currentMapColorScale;
            if (!scale) {
                const countryValues = filteredData.map((c) =>
                    resolveCountryMetricValue(c, metric, isCumulative)
                );
                scale = buildColorScaleFromValues(countryValues, metric, isCumulative);
                currentMapColorScale = scale;
            }
            updateMapLegend(
                scale,
                true,
                `Countries: ${selectedValueTypes.map((vt) => formatValueType(vt)).join(', ')}`
            );
            
            // Create lookup map by country name
            const countryMetrics = {};
            filteredData.forEach(c => {
                const countryName = c.country;
                // Store by multiple possible keys for flexible matching
                countryMetrics[countryName] = {
                    value_loss: isCumulative ? (c.cumulative_loss || 0) : (c.value_loss || 0),
                    loss_fraction: c.loss_fraction || 0,
                    original_value: c.original_value || 0,
                    iso_a3: c.iso_a3 || ''
                };
                // Also try common name variations
                if (countryName.includes(',')) {
                    countryMetrics[countryName.split(',')[0].trim()] = countryMetrics[countryName];
                }
            });
            
            // Remove existing choropleth
            if (choroplethLayer) {
                map.removeLayer(choroplethLayer);
            }
            choroplethLayer = L.layerGroup();
            
            // Create styled GeoJSON layer
            const styledGeoJson = L.geoJSON(countryBoundaries, {
                pane: 'choroplethPane',
                style: (feature) => {
                    const countryName = feature.properties.name;
                    // Try exact match first, then try partial matches
                    let countryMetric = countryMetrics[countryName];
                    if (!countryMetric) {
                        // Try matching by partial name
                        for (const [key, value] of Object.entries(countryMetrics)) {
                            if (countryName.includes(key) || key.includes(countryName)) {
                                countryMetric = value;
                                break;
                            }
                        }
                    }
                    
                    // Determine color value based on metric
                    let colorValue = null;
                    if (countryMetric) {
                        colorValue = resolveFeatureMetricValue(countryMetric, metric, isCumulative);
                    }
                    
                    const fillColor = colorValue !== null ? getColorFromScale(colorValue, scale) : '#64748b';
                    
                    return {
                        fillColor: fillColor,
                        weight: 1,
                        opacity: 0.8,
                        color: '#1a2332',
                        fillOpacity: countryMetric ? 0.6 : 0,
                        dashArray: countryMetric ? null : '5, 5'
                    };
                },
                onEachFeature: (feature, layer) => {
                    const countryName = feature.properties.name;
                    // Try exact match first, then try partial matches
                    let countryMetric = countryMetrics[countryName];
                    if (!countryMetric) {
                        for (const [key, value] of Object.entries(countryMetrics)) {
                            if (countryName.includes(key) || key.includes(countryName)) {
                                countryMetric = value;
                                break;
                            }
                        }
                    }
                    
                    if (countryMetric) {
                        const lossType = isCumulative ? 'Cumulative' : 'Annual';
                        const formatMoney = (val) => {
                            if (val >= 1e9) return `$${(val / 1e9).toFixed(2)}B`;
                            if (val >= 1e6) return `$${(val / 1e6).toFixed(1)}M`;
                            if (val >= 1e3) return `$${(val / 1e3).toFixed(1)}k`;
                            return `$${val.toFixed(0)}`;
                        };
                        
                        const popupContent = `
                            <div class="popup-content">
                                <strong>${countryName}</strong><br>
                                ${lossType} Loss: ${formatMoney(countryMetric.value_loss)}<br>
                                Loss: ${(countryMetric.loss_fraction * 100).toFixed(1)}%
                            </div>
                        `;
                        layer.bindPopup(popupContent);
                    }
                }
            });
            
            choroplethLayer.addLayer(styledGeoJson);
            choroplethLayer.addTo(map);
            // Reassert point precedence after choropleth draw.
            bringTourismLayersToFront();
            
            // Add or update legend control
            addChoroplethLegend();
        }
        
        let choroplethLegendControl = null;
        
        function addChoroplethLegend() {
            // Remove existing legend if any
            if (choroplethLegendControl) {
                map.removeControl(choroplethLegendControl);
            }
            
            choroplethLegendControl = L.control({position: 'bottomright'});
            
            choroplethLegendControl.onAdd = function(map) {
                const div = L.DomUtil.create('div', 'choropleth-legend');
                div.style.cssText = 'background: none; padding: 0; border: none; font-family: Instrument Sans, sans-serif; font-size: 12px; color: #64748b;';
                
                // Only the text at the bottom (no colored boxes or label)
                div.innerHTML = '<div style="color: #64748b; font-size: 11px; padding: 0.25em 0.5em; background: none;">Dashed = No data</div>';
                
                return div;
            };
            
            choroplethLegendControl.addTo(map);
        }
        
        function removeChoroplethLegend() {
            if (choroplethLegendControl) {
                map.removeControl(choroplethLegendControl);
                choroplethLegendControl = null;
            }
        }

        // ============================================================
        // INITIALIZE
        // ============================================================
        
        document.addEventListener('DOMContentLoaded', loadData);
