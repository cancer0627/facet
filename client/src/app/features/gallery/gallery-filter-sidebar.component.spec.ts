import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MatDialog } from '@angular/material/dialog';
import { provideNativeDateAdapter } from '@angular/material/core';
import { GalleryStore } from './gallery.store';
import { ADDITIONAL_FILTERS, GalleryFilterSidebarComponent } from './gallery-filter-sidebar.component';
import { I18nService } from '../../core/services/i18n.service';
import { AuthService } from '../../core/services/auth.service';
import { AlbumService } from '../../core/services/album.service';

describe('GalleryFilterSidebarComponent', () => {
  let component: GalleryFilterSidebarComponent;
  let mockStore: Record<string, unknown>;

  beforeEach(() => {
    mockStore = {
      filters: signal({
        hide_details: true, hide_blinks: true, hide_bursts: true, hide_duplicates: true,
        hide_rejected: true, favorites_only: false, is_monochrome: false,
        camera: '', lens: '', tag: '', composition_pattern: '', person_id: '',
        min_score: '', max_score: '', min_aesthetic: '', max_aesthetic: '',
        min_face_quality: '', max_face_quality: '', min_composition: '', max_composition: '',
        min_sharpness: '', max_sharpness: '', min_exposure: '', max_exposure: '',
        min_color: '', max_color: '', min_contrast: '', max_contrast: '',
        min_noise: '', max_noise: '', min_dynamic_range: '', max_dynamic_range: '',
        min_face_count: '', max_face_count: '', min_eye_sharpness: '', max_eye_sharpness: '',
        min_face_sharpness: '', max_face_sharpness: '', min_face_ratio: '', max_face_ratio: '',
        min_face_confidence: '', max_face_confidence: '', min_quality_score: '', max_quality_score: '',
        min_topiq: '', max_topiq: '', min_power_point: '', max_power_point: '',
        min_leading_lines: '', max_leading_lines: '', min_isolation: '', max_isolation: '',
        min_saturation: '', max_saturation: '', min_luminance: '', max_luminance: '',
        min_histogram_spread: '', max_histogram_spread: '', min_star_rating: '', max_star_rating: '',
        min_iso: '', max_iso: '', min_aperture: '', max_aperture: '',
        min_focal_length: '', max_focal_length: '', date_from: '', date_to: '',
        search: '', type: '', sort: 'aggregate', sort_direction: 'DESC', page: 1, per_page: 64,
        similar_to: '', similarity_mode: 'visual', min_similarity: '70',
        semanticQuery: '', search_threshold: '', album_id: '',
        min_aesthetic_iaa: '', max_aesthetic_iaa: '',
        min_face_quality_iqa: '', max_face_quality_iqa: '',
        min_liqe: '', max_liqe: '',
        min_qrealign: '', max_qrealign: '',
        min_aesthetic_v25: '', max_aesthetic_v25: '',
        min_deqa: '', max_deqa: '',
        min_subject_sharpness: '', max_subject_sharpness: '',
        min_subject_prominence: '', max_subject_prominence: '',
        min_subject_placement: '', max_subject_placement: '',
        min_bg_separation: '', max_bg_separation: '',
        path_prefix: '',
      }),
      filterDrawerOpen: signal(true),
      cameras: signal([]),
      lenses: signal([]),
      tags: signal([]),
      persons: signal([]),
      compositionPatterns: signal([]),
      metricRanges: signal({}),
      config: signal(null),
      activeFilterCount: signal(0),
      updateFilter: vi.fn(),
      updateFilterDebounced: vi.fn(),
      updateFilters: vi.fn(),
      resetFilters: vi.fn(),
      setFilterDrawerOpen: vi.fn(),
    };

    TestBed.configureTestingModule({
      providers: [
        GalleryFilterSidebarComponent,
        { provide: GalleryStore, useValue: mockStore },
        { provide: I18nService, useValue: { t: vi.fn((k: string) => k), currentLang: vi.fn(() => 'en'), translations: vi.fn(() => ({})) } },
        { provide: AuthService, useValue: { isEdition: vi.fn(() => false) } },
        { provide: AlbumService, useValue: { list: vi.fn(() => of({ albums: [] })) } },
        { provide: MatDialog, useValue: { open: vi.fn() } },
      ],
    });
    component = TestBed.inject(GalleryFilterSidebarComponent);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('shows only aggregate and aesthetic ranges in the quality section', () => {
    const qualityFilterIds = ADDITIONAL_FILTERS
      .filter(def => def.sectionKey === 'gallery.sidebar.quality')
      .map(def => def.id);

    expect(qualityFilterIds).toEqual(['score_range', 'aesthetic_range']);
  });

  describe('clampDef', () => {
    const scoreDef = {
      id: 'score_range', minKey: 'min_score', maxKey: 'max_score',
      sliderMin: 0, sliderMax: 10, step: 0.5,
    };

    it('clamps slider bounds to the data-driven range when no value is out of range', () => {
      const store = (component as any).store;
      store.metricRanges.set({ min_score: { min: 5, max: 9, buckets: [] } });
      store.filters.set({ ...store.filters(), min_score: '', max_score: '' });
      const clamped = (component as any).clampDef(scoreDef);
      expect(clamped.sliderMin).toBe(5);
      expect(clamped.sliderMax).toBe(9);
    });

    it('widens bounds so an active value below the data min stays reachable', () => {
      const store = (component as any).store;
      store.metricRanges.set({ min_score: { min: 5, max: 9, buckets: [] } });
      // A persisted/URL min of 2 sits below the data minimum of 5.
      store.filters.set({ ...store.filters(), min_score: '2', max_score: '' });
      const clamped = (component as any).clampDef(scoreDef);
      expect(clamped.sliderMin).toBeLessThanOrEqual(2); // not pinned to 5 — reachable
      expect(clamped.sliderMax).toBe(9);                // data max unchanged
    });
  });

  describe('sectionActiveCounts', () => {
    it('returns 0 for all sections when no filters are active', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), hide_rejected: false });
      const counts = component.sectionActiveCounts();
      expect(counts['date']).toBe(0);
      expect(counts['content']).toBe(0);
      expect(counts['equipment']).toBe(0);
      expect(counts['refine']).toBe(0);
      expect(counts['gallery.sidebar.quality']).toBe(0);
      expect(counts['gallery.sidebar.face']).toBe(0);
    });

    it('counts date_from and date_to independently', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), date_from: '2024-01-01' });
      expect(component.sectionActiveCounts()['date']).toBe(1);
      mockStore.filters.set({ ...mockStore.filters(), date_to: '2024-12-31' });
      expect(component.sectionActiveCounts()['date']).toBe(2);
    });

    it('counts tag and composition_pattern for content section', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), tag: 'landscape' });
      expect(component.sectionActiveCounts()['content']).toBe(1);
      mockStore.filters.set({ ...mockStore.filters(), composition_pattern: 'rule_of_thirds' });
      expect(component.sectionActiveCounts()['content']).toBe(2);
    });

    it('counts camera and lens for equipment section', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), camera: 'Sony A7', lens: 'FE 85mm' });
      expect(component.sectionActiveCounts()['equipment']).toBe(2);
    });

    it('counts an active path_prefix for the folder section', () => {
      const mockStore = (component as any).store;
      expect(component.sectionActiveCounts()['folder']).toBe(0);
      mockStore.filters.set({ ...mockStore.filters(), path_prefix: '/photos/Family/' });
      expect(component.sectionActiveCounts()['folder']).toBe(1);
    });

    it('counts favorites_only, is_monochrome, hide_rejected for refine section', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), favorites_only: true, is_monochrome: true, hide_rejected: true });
      expect(component.sectionActiveCounts()['refine']).toBe(3);
    });

    it('counts active metric filters by min/max key', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), min_score: '7', min_aesthetic: '6' });
      expect(component.sectionActiveCounts()['gallery.sidebar.quality']).toBe(2);
    });

    it('counts a metric filter once even when both min and max are set', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), min_score: '5', max_score: '9' });
      expect(component.sectionActiveCounts()['gallery.sidebar.quality']).toBe(1);
    });
  });

  describe('onDynamicRangeChange', () => {
    const iaaFilter = {
      id: 'aesthetic_iaa_range', labelKey: 'gallery.aesthetic_iaa_range',
      sectionKey: 'gallery.sidebar.extended_quality',
      minKey: 'min_aesthetic_iaa' as const, maxKey: 'max_aesthetic_iaa' as const,
      sliderMin: 0, sliderMax: 10, step: 0.5, spanWidth: 'w-16',
    };

    const isoFilter = {
      id: 'iso_range', labelKey: 'gallery.iso_range',
      sectionKey: 'gallery.sidebar.exposure_range',
      minKey: 'min_iso' as const, maxKey: 'max_iso' as const,
      sliderMin: 50, sliderMax: 25600, step: 50, spanWidth: 'w-20',
    };

    it('clears min filter at slider minimum', () => {
      const mockStore = (component as any).store;
      component.onDynamicRangeChange(iaaFilter, 'min', 0);
      expect(mockStore.updateFilterDebounced).toHaveBeenCalledWith('min_aesthetic_iaa', '');
    });

    it('redirects max to min when min is at default', () => {
      const mockStore = (component as any).store;
      component.onDynamicRangeChange(iaaFilter, 'max', 7.5);
      expect(mockStore.updateFilterDebounced).toHaveBeenCalledWith('min_aesthetic_iaa', '7.5');
    });

    it('clears max filter at slider maximum when min is set', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), min_aesthetic_iaa: '5' });
      component.onDynamicRangeChange(iaaFilter, 'max', 10);
      expect(mockStore.updateFilterDebounced).toHaveBeenCalledWith('max_aesthetic_iaa', '');
    });

    it('stores string value for non-boundary min', () => {
      const mockStore = (component as any).store;
      component.onDynamicRangeChange(iaaFilter, 'min', 6.5);
      expect(mockStore.updateFilterDebounced).toHaveBeenCalledWith('min_aesthetic_iaa', '6.5');
    });

    it('stores string value for non-boundary max when min is set', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), min_aesthetic_iaa: '3' });
      component.onDynamicRangeChange(iaaFilter, 'max', 7.5);
      expect(mockStore.updateFilterDebounced).toHaveBeenCalledWith('max_aesthetic_iaa', '7.5');
    });

    it('clears min at non-zero boundary (ISO 50)', () => {
      const mockStore = (component as any).store;
      component.onDynamicRangeChange(isoFilter, 'min', 50);
      expect(mockStore.updateFilterDebounced).toHaveBeenCalledWith('min_iso', '');
    });

    it('clears max at non-standard boundary (ISO 25600) when min is set', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), min_iso: '100' });
      component.onDynamicRangeChange(isoFilter, 'max', 25600);
      expect(mockStore.updateFilterDebounced).toHaveBeenCalledWith('max_iso', '');
    });
  });

  describe('folder filter', () => {
    it('clearFolderFilter empties path_prefix', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), path_prefix: '/photos/Family/' });
      component.clearFolderFilter();
      expect(mockStore.updateFilter).toHaveBeenCalledWith('path_prefix', '');
    });

    it('currentFolderName shows the last segment of the active prefix', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), path_prefix: '/photos/Family/2026/' });
      expect(component.currentFolderName()).toBe('2026');
    });
  });

  describe('semantic search threshold slider', () => {
    // Note: this suite tests the component's methods/computeds directly, the
    // same style every other describe block in this file uses -- the harness
    // never renders the template via fixture.createComponent/detectChanges,
    // so panel-visibility (@if show_semantic_search, pre-existing/unchanged)
    // is not exercised here either.
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('searchThresholdValue mirrors the string filter as a number for the slider [value] binding', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), search_threshold: '12' });
      expect((component as any).searchThresholdValue()).toBe(12);
    });

    it('searchThresholdValue falls back to 0 when unseeded and config has not loaded', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), search_threshold: '' });
      mockStore.config.set(null);
      expect((component as any).searchThresholdValue()).toBe(0);
    });

    it('searchThresholdValue falls back to the config default (as a percent) when unseeded -- never a bare 0/blank readout while the server actually gates at its own default', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), search_threshold: '' });
      mockStore.config.set({ search_threshold_default: 0.05 });
      expect((component as any).searchThresholdValue()).toBe(5);
    });

    it('onSearchThresholdChange debounces the filter update by 400ms', () => {
      const mockStore = (component as any).store;
      component.onSearchThresholdChange(20);
      expect(mockStore.updateFilter).not.toHaveBeenCalled();

      vi.advanceTimersByTime(400);

      expect(mockStore.updateFilter).toHaveBeenCalledWith('search_threshold', '20');
    });

    it('the readout follows the drag immediately, rather than freezing for the debounce', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), search_threshold: '12' });

      component.onSearchThresholdChange(31);

      expect((component as any).searchThresholdValue()).toBe(31);
      expect(mockStore.updateFilter).not.toHaveBeenCalled();
    });

    it('drops the draft once the debounce fires, so the filter is the single source again', () => {
      const mockStore = (component as any).store;
      mockStore.filters.set({ ...mockStore.filters(), search_threshold: '12' });
      component.onSearchThresholdChange(31);

      vi.advanceTimersByTime(400);
      // The mock store never applies updateFilter, so the filter is still '12':
      // a draft that outlived the timeout would keep reading 31.
      expect((component as any).searchThresholdValue()).toBe(12);
    });

    it('a second drag replaces the draft rather than queueing a stale readout', () => {
      component.onSearchThresholdChange(31);
      vi.advanceTimersByTime(200);
      component.onSearchThresholdChange(44);

      expect((component as any).searchThresholdValue()).toBe(44);

      vi.advanceTimersByTime(400);

      expect((component as any).store.updateFilter).toHaveBeenCalledTimes(1);
      expect((component as any).store.updateFilter).toHaveBeenCalledWith('search_threshold', '44');
    });

    it('uses its own debounce timer, independent of the semantic-query search timer', () => {
      const mockStore = (component as any).store;
      const inputEvent = { target: { value: 'sunset' } } as unknown as Event;

      component.onSemanticSearch(inputEvent);
      component.onSearchThresholdChange(30);
      vi.advanceTimersByTime(400);

      expect(mockStore.updateFilter).toHaveBeenCalledWith('semanticQuery', 'sunset');
      expect(mockStore.updateFilter).toHaveBeenCalledWith('search_threshold', '30');
    });
  });

  describe('semantic search threshold slider (rendered)', () => {
    // The only block in this file that renders the template: the slider is Material
    // markup, so [value]/(valueChange) wiring and the aria attributes cannot be
    // reached through the component's methods alone.
    let fixture: ComponentFixture<GalleryFilterSidebarComponent>;
    let thumb: HTMLInputElement;

    beforeEach(() => {
      TestBed.resetTestingModule();
      const renderedStore = {
        ...mockStore,
        config: signal({ features: { show_semantic_search: true }, search_threshold_default: 0.05 }),
        cardWidth: signal(220),
        colorTemps: signal([]),
        hueBuckets: signal([]),
        patterns: signal([]),
        types: signal([]),
        currentAlbum: signal(null),
        galleryMode: signal('grid'),
        gpsLocationName: signal(''),
        virtualScroll: signal(false),
        setCardWidth: vi.fn(),
        setGalleryMode: vi.fn(),
        setVirtualScroll: vi.fn(),
      };

      TestBed.configureTestingModule({
        providers: [
          provideNativeDateAdapter(),
          { provide: GalleryStore, useValue: renderedStore },
          { provide: I18nService, useValue: { t: vi.fn((k: string) => k), currentLang: vi.fn(() => 'en'), translations: vi.fn(() => ({})) } },
          { provide: AuthService, useValue: { isEdition: vi.fn(() => false) } },
          { provide: AlbumService, useValue: { list: vi.fn(() => of({ albums: [] })) } },
          { provide: MatDialog, useValue: { open: vi.fn() } },
        ],
      });

      fixture = TestBed.createComponent(GalleryFilterSidebarComponent);
      fixture.detectChanges();
      thumb = fixture.nativeElement.querySelector('mat-slider input');
    });

    it('renders the thumb seeded from the config default, announced with its unit', () => {
      expect(thumb).toBeTruthy();
      expect(thumb.value).toBe('5');
      expect(thumb.getAttribute('aria-valuetext')).toBe('5%');
      expect(thumb.getAttribute('aria-describedby')).toBe('semantic-search-threshold-hint');
      expect(fixture.nativeElement.querySelector('#semantic-search-threshold-hint')).toBeTruthy();
    });

    it('wires the thumb back to the component, readout and valuetext following the drag', () => {
      thumb.value = '31';
      thumb.dispatchEvent(new Event('input'));
      thumb.dispatchEvent(new Event('change'));
      fixture.detectChanges();

      expect(fixture.nativeElement.textContent).toContain('31%');
      expect(thumb.getAttribute('aria-valuetext')).toBe('31%');
    });
  });
});
