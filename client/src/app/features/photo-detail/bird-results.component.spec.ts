import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { BirdResultsComponent } from './bird-results.component';

describe('BirdResultsComponent', () => {
  const result = {
    status: 'scored', score: 7.68, count: 2, best_index: 0,
    stale: false, outdated: false, error: null,
    assessment: 'pass', assessment_reason: 'complete_and_clear', assessment_index: 0,
    focus_score: 7.68, focus_score_index: 0,
    birds: [
      { sharpness: 7.68, confidence: .9, width: 400, height: 300, assessment: 'pass', focus_score: 7.68 },
      { sharpness: null, confidence: .8, width: 300, height: 200, reason: 'truncated', assessment: 'review' },
    ],
  };

  async function build(overrides = {}, species: 'bird' | 'elephant' = 'bird') {
    TestBed.configureTestingModule({
      imports: [BirdResultsComponent],
      providers: [
        { provide: ApiService, useValue: { get: vi.fn(() => of({ ...result, ...overrides })) } },
        { provide: I18nService, useValue: { translations: () => ({}), t: (key: string) => key } },
      ],
    });
    const fixture = TestBed.createComponent(BirdResultsComponent);
    fixture.componentRef.setInput('path', '/a & b.jpg');
    fixture.componentRef.setInput('species', species);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    return fixture;
  }

  it('loads and previews elephant evidence independently', async () => {
    const fixture = await build({ elephants: result.birds }, 'elephant');
    const api = TestBed.inject(ApiService);
    expect(api.get).toHaveBeenCalledWith('/elephants/photo', { path: '/a & b.jpg' });
    const element = fixture.nativeElement as HTMLElement;
    expect(element.textContent).toContain('elephants.title');
    expect(element.textContent).not.toContain('birds.title');
    element.querySelector('button')!.click();
    fixture.detectChanges();
    expect(element.querySelectorAll('img')[1].getAttribute('src')).toContain('/api/elephants/preview?');
  });

  it('opens bird zero and lets the user inspect an unscored bird with its reason', async () => {
    const fixture = await build();
    const element = fixture.nativeElement as HTMLElement;
    element.querySelector('button')!.click();
    fixture.detectChanges();
    const images = element.querySelectorAll('img');
    expect(images[1].getAttribute('src')).toContain('index=0');
    expect(images[1].getAttribute('src')).toContain('path=%2Fa+%26+b.jpg');
    const select = element.querySelector('select')!;
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(element.querySelectorAll('img')[1].getAttribute('src')).toContain('index=1');
    expect(element.textContent).toContain('birds.reason.truncated');
  });

  it('does not display per-bird scores from an outdated result', async () => {
    const fixture = await build({ outdated: true, score: null });
    const element = fixture.nativeElement as HTMLElement;
    element.querySelector('button')!.click();
    fixture.detectChanges();
    expect(element.textContent).toContain('birds.outdated');
    expect(element.textContent).not.toContain('7.68');
    expect(element.querySelector('select')).not.toBeNull();
  });

  it('selects the best bird after asynchronously created options appear', async () => {
    const fixture = await build({ best_index: 1, assessment_index: 1, focus_score_index: 1, birds: [result.birds[1], result.birds[0]] });
    const element = fixture.nativeElement as HTMLElement;
    element.querySelector('button')!.click();
    fixture.detectChanges();
    expect(element.querySelector('select')!.value).toBe('1');
    expect(element.querySelectorAll('img')[1].getAttribute('src')).toContain('index=1');
  });

  it('resets crop selection when the displayed photo changes', async () => {
    const fixture = await build();
    const element = fixture.nativeElement as HTMLElement;
    element.querySelector('button')!.click();
    fixture.detectChanges();
    expect(element.querySelector('select')).not.toBeNull();
    fixture.componentRef.setInput('path', '/next.jpg');
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    expect(element.querySelector('select')).toBeNull();
  });

  it('explains why a tiny sharp bird is excluded without displaying a reference score', async () => {
    const fixture = await build({ min_area_percent: 1, excluded_small_count: 1, birds: [result.birds[0], {
      sharpness: 9.3, confidence: .9, width: 87, height: 133, area_percent: .027,
      exclusion_reason: 'below_min_area', reason: null,
    }] });
    const element = fixture.nativeElement as HTMLElement;
    expect(element.textContent).toContain('birds.size_rule');
    expect(element.textContent).toContain('birds.small_excluded');
    element.querySelector('button')!.click();
    fixture.detectChanges();
    const select = element.querySelector('select')!;
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(element.textContent).toContain('birds.reason.below_min_area');
    expect(element.textContent).toContain('birds.size_metric');
    expect(select.options[1].textContent).toContain('birds.assessment.review');
    expect(element.textContent).not.toContain('9.30');
  });

  it('labels unsupported focus estimates for review instead of displaying a high grade', async () => {
    const fixture = await build({ focus_unverified_count: 1, birds: [result.birds[0], {
      sharpness: 9.3, confidence: .99, width: 1200, height: 1800, focus_verified: false,
      exclusion_reason: 'focus_unverified', reason: null,
    }] });
    const element = fixture.nativeElement as HTMLElement;
    expect(element.textContent).toContain('birds.focus_unverified_count');
    element.querySelector('button')!.click();
    fixture.detectChanges();
    const select = element.querySelector('select')!;
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(select.options[1].textContent).toContain('birds.assessment.review');
    expect(element.textContent).not.toContain('9.30');
    expect(element.textContent).toContain('birds.reason.focus_unverified');
  });

  it('labels a sharp frame fragment for review without presenting it as a high grade', async () => {
    const fixture = await build({ birds: [result.birds[0], {
      sharpness: 9.3, confidence: .99, width: 2331, height: 1058, focus_verified: true,
      exclusion_reason: 'frame_edge', reason: null,
    }] });
    const element = fixture.nativeElement as HTMLElement;
    element.querySelector('button')!.click();
    fixture.detectChanges();
    const select = element.querySelector('select')!;
    select.value = '1';
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(select.options[1].textContent).toContain('birds.assessment.review');
    expect(element.textContent).not.toContain('9.30');
    expect(element.textContent).toContain('birds.reason.frame_edge');
  });

  it('shows the verified head-eye score on a ten-point scale without a scale suffix instead of the stored whole-bird score', async () => {
    const fixture = await build({ score: 9.3, focus_score: 2, assessment: 'fail', assessment_reason: 'head_eye_blurred' });
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('birds.focus_score: 2.0');
    expect(text).not.toContain('birds.assessment.fail');
    expect(text).not.toContain('9.30');
    expect(text).not.toMatch(/\/ 10(?:0)?\b/);
  });

  it('leaves an unmeasurable eye score blank instead of inventing a middle score', async () => {
    const fixture = await build({ focus_score: null, assessment: 'review', assessment_reason: 'head_eye_uncertain' });
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('birds.assessment.review');
    expect(text).toContain('birds.assessment_reason.head_eye_uncertain');
    expect(text).not.toContain('/ 100');
  });
});
