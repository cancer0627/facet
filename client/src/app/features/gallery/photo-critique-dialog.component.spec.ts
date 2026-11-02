import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { I18nService } from '../../core/services/i18n.service';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import { MismatchReasonPipe, PhotoCritiqueDialogComponent } from './photo-critique-dialog.component';
import { NEVER, Subject, of, throwError } from 'rxjs';

describe('MismatchReasonPipe', () => {
  let pipe: MismatchReasonPipe;
  let mockI18n: { t: Mock };

  beforeEach(() => {
    mockI18n = { t: vi.fn((key: string) => key) };

    TestBed.configureTestingModule({
      providers: [
        { provide: I18nService, useValue: mockI18n },
      ],
    });

    pipe = TestBed.runInInjectionContext(() => new MismatchReasonPipe());
  });

  describe('required_tags', () => {
    it('formats required tags up to 3', () => {
      pipe.transform({ key: 'required_tags', required: ['landscape', 'mountain'], actual: [] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.required_tags', { tags: 'landscape, mountain' });
    });

    it('truncates with ellipsis when more than 3 tags', () => {
      pipe.transform({ key: 'required_tags', required: ['a', 'b', 'c', 'd'], actual: [] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.required_tags', { tags: 'a, b, c, …' });
    });

    it('handles empty required array', () => {
      pipe.transform({ key: 'required_tags', required: [], actual: [] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.required_tags', { tags: '' });
    });
  });

  describe('excluded_tags', () => {
    it('formats matched excluded tags', () => {
      pipe.transform({ key: 'excluded_tags', required: ['indoor'], actual: ['indoor'] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.excluded_tags', { tags: 'indoor' });
    });

    it('joins multiple excluded tags', () => {
      pipe.transform({ key: 'excluded_tags', required: ['indoor', 'text'], actual: ['indoor', 'text'] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.excluded_tags', { tags: 'indoor, text' });
    });
  });

  describe('boolean filters', () => {
    it('uses base key when required is true', () => {
      pipe.transform({ key: 'has_face', required: true, actual: false });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.has_face');
    });

    it('uses _false suffix when required is false', () => {
      pipe.transform({ key: 'is_monochrome', required: false, actual: true });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.is_monochrome_false');
    });

    it('handles is_silhouette', () => {
      pipe.transform({ key: 'is_silhouette', required: true, actual: false });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.is_silhouette');
    });

    it('handles is_group_portrait', () => {
      pipe.transform({ key: 'is_group_portrait', required: false, actual: true });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.is_group_portrait_false');
    });
  });

  describe('numeric filters', () => {
    it('reports no_value when actual is null', () => {
      pipe.transform({ key: 'face_ratio_min', required: 0.05, actual: null });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.no_value');
    });

    it('reports no_value when actual is undefined', () => {
      pipe.transform({ key: 'iso_max', required: 6400, actual: undefined });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.no_value');
    });

    it('formats numeric mismatch with required and actual', () => {
      pipe.transform({ key: 'face_ratio_min', required: 0.05, actual: 0.02 });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.face_ratio_min', {
        required: '0.05',
        actual: '0.02',
      });
    });

    it('handles zero actual', () => {
      pipe.transform({ key: 'face_count_min', required: 1, actual: 0 });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.face_count_min', {
        required: '1',
        actual: '0',
      });
    });
  });
});

describe('PhotoCritiqueDialogComponent overlay error handling', () => {
  it('reverts the overlay toggle and notifies when the overlay image fails to load', () => {
    const mockApi = { get: vi.fn() };
    const mockAuth = { hasFeature: vi.fn(() => true) };
    const mockI18n = { t: vi.fn((key: string) => key), locale: vi.fn(() => 'en') };
    const mockSnack = { open: vi.fn() };

    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [
        PhotoCritiqueDialogComponent,
        { provide: ApiService, useValue: mockApi },
        { provide: AuthService, useValue: mockAuth },
        { provide: I18nService, useValue: mockI18n },
        { provide: MatSnackBar, useValue: mockSnack },
        { provide: MAT_DIALOG_DATA, useValue: { photoPath: '/p/x.jpg', vlmAvailable: false } },
      ],
    });
    const component: any = TestBed.inject(PhotoCritiqueDialogComponent);
    component.overlayOn.set(true);

    component.onOverlayError();

    expect(component.overlayOn()).toBe(false);
    expect(mockSnack.open).toHaveBeenCalled();
  });
});

describe('PhotoCritiqueDialogComponent personalized suggestions', () => {
  function setup(mockApi: { get: Mock }) {
    const mockAuth = { hasFeature: vi.fn(() => true) };
    const mockI18n = { t: vi.fn((key: string) => key), locale: signal('zh') };
    const mockSnack = { open: vi.fn() };

    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [
        PhotoCritiqueDialogComponent,
        { provide: ApiService, useValue: mockApi },
        { provide: AuthService, useValue: mockAuth },
        { provide: I18nService, useValue: mockI18n },
        { provide: MatSnackBar, useValue: mockSnack },
        { provide: MAT_DIALOG_DATA, useValue: { photoPath: '/p/x.jpg', vlmAvailable: false } },
      ],
    });
    return { component: TestBed.inject(PhotoCritiqueDialogComponent) as any, mockSnack, mockI18n };
  }

  const critique = {
    category: 'landscape',
    category_reason: { reason_key: 'default', category: 'landscape', details: [] },
    aggregate: 7.5,
    breakdown: [],
    strengths: [],
    weaknesses: [],
    suggestions: [],
    penalties: {},
  };

  const personalized = {
    available: true,
    source: 'cached' as const,
    aggregate_score: 7.5,
    vcg_submission_score: 7.15,
    aggregate_suggestions: [{ action: '拍摄：改变机位', reason: '主体边缘杂乱' }],
    vcg_suggestions: [{ action: '后期：压低高光', reason: '高光区域过亮' }],
    generated_at: '2026-09-28T00:00:00+00:00',
    reason: null,
    warning: null,
  };

  it('shows rules while combined generation is pending and then reads cached suggestions', async () => {
    const generated = new Subject<unknown>();
    const mockApi = { get: vi.fn((path: string, params: Record<string, string>) => {
      if (path === '/personalized_suggestions') return of(personalized);
      return params['mode'] === 'combined' ? generated : of(critique);
    }) };
    const { component } = setup(mockApi);
    component.data.vlmAvailable = true;
    const pending = component.ngOnInit();
    await Promise.resolve();
    expect(component.critique()).toEqual(critique);
    expect(component.loading()).toBe(false);
    expect(mockApi.get).not.toHaveBeenCalledWith('/personalized_suggestions', expect.anything());
    generated.next({ ...critique, vlm_critique: '观察：山景。' });
    await pending;
    await Promise.resolve();
    expect(component.critique().vlm_critique).toBe('观察：山景。');
    expect(component.personalizedSuggestions()).toEqual(personalized);
  });

  it('reloads in the selected language and ignores late previous-language responses', async () => {
    const pending = new Map<string, Subject<unknown>>();
    const mockApi = {
      get: vi.fn((path: string, params: Record<string, string>) => {
        const response = new Subject<unknown>();
        pending.set(`${path}:${params['lang']}`, response);
        return response;
      }),
    };
    const { component, mockI18n } = setup(mockApi);
    const initial = component.ngOnInit();
    TestBed.tick();
    mockI18n.locale.set('en');
    TestBed.tick();
    pending.get('/critique:en')!.next({ ...critique, vlm_critique: 'English critique' });
    pending.get('/personalized_suggestions:en')!.next({ ...personalized, lang: 'en' });
    await Promise.resolve();
    pending.get('/critique:zh')!.next({ ...critique, vlm_critique: '旧中文评价' });
    pending.get('/personalized_suggestions:zh')!.error(new Error('late error'));
    await initial;
    expect(component.critique().vlm_critique).toBe('English critique');
    expect(component.personalizedSuggestions().lang).toBe('en');
    expect(component.personalizedError()).toBeNull();
    expect(component.loading()).toBe(false);
  });

  it('clears visible previous-language content while the new language loads', async () => {
    const mockApi = { get: vi.fn(() => of(critique)) };
    const { component, mockI18n } = setup(mockApi);
    await component.ngOnInit();
    TestBed.tick();
    mockApi.get.mockImplementation(() => NEVER);
    mockI18n.locale.set('fr');
    TestBed.tick();
    expect(component.critique()).toBeNull();
    expect(component.personalizedSuggestions()).toBeNull();
    expect(component.loading()).toBe(true);
  });

  it('loads both personalized groups independently and refreshes only that endpoint', async () => {
    const mockApi = {
      get: vi.fn((path: string) => path === '/critique' ? of(critique) : of(personalized)),
    };
    const { component } = setup(mockApi);

    await component.ngOnInit();
    await new Promise(resolve => setTimeout(resolve, 0));

    expect(component.personalizedSuggestions()).toEqual(personalized);
    expect(mockApi.get).toHaveBeenCalledWith('/personalized_suggestions', expect.objectContaining({
      path: '/p/x.jpg', lang: 'zh', refresh: 'false',
    }));

    await component.refreshPersonalized();
    expect(mockApi.get).toHaveBeenLastCalledWith('/personalized_suggestions', expect.objectContaining({
      refresh: 'true',
    }));
    expect(mockApi.get).not.toHaveBeenCalledWith('/critique', expect.objectContaining({ refresh: 'true' }));
  });

  it('keeps the previous content when refresh fails', async () => {
    const mockApi = {
      get: vi.fn((path: string, params: Record<string, string>) => {
        if (path === '/critique') return of(critique);
        return params['refresh'] === 'true' ? throwError(() => new Error('refresh failed')) : of(personalized);
      }),
    };
    const { component, mockSnack } = setup(mockApi);

    await component.ngOnInit();
    await new Promise(resolve => setTimeout(resolve, 0));
    await component.refreshPersonalized();
    await new Promise(resolve => setTimeout(resolve, 0));

    expect(component.personalizedSuggestions()).toEqual(personalized);
    expect(mockSnack.open).toHaveBeenCalled();
  });

  it('ends the loading state when the personalized request times out', async () => {
    vi.useFakeTimers();
    try {
      const mockApi = {
        get: vi.fn((path: string) => path === '/critique' ? of(critique) : NEVER),
      };
      const { component } = setup(mockApi);

      await component.ngOnInit();
      await vi.advanceTimersByTimeAsync(120_000);

      expect(component.personalizedLoading()).toBe(false);
      expect(component.personalizedError()).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
  });
});
